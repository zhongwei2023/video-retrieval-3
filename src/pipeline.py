from __future__ import annotations

import json
import time
import gc
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_target_extractor import parse_target_with_llm
from .video_sampler import sample_frames, FrameSample
from .clip_retriever import CLIPRetriever
from .text_localizer import OwlV2Localizer, OWLConfig
from .frame_scorer import score_detection, laplacian_sharpness
from .cropper import crop_with_padding, save_outputs
from .detection import Detection


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass
class PipelineConfig:
    video_path: str
    query: str
    out_dir: str
    fps: Optional[float] = None
    sampling_mode: str = "high_density"
    max_side: int = 1024
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    crop_padding_ratio: float = 0.08
    clip_topk: int = 30
    clip_batch_size: int = 32
    owl_batch_size: int = 1


def _run_owlv2_stage(
    candidate_frames: List[FrameSample],
    detect_query: str,
    box_threshold: float,
    text_threshold: float,
    owl_batch_size: int,
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = time.time()
    localizer = OwlV2Localizer(
        OWLConfig(box_threshold=box_threshold, text_threshold=text_threshold)
    )
    candidates: List[Dict[str, Any]] = []
    bs = max(1, owl_batch_size)

    for bi in range(0, len(candidate_frames), bs):
        batch = candidate_frames[bi : bi + bs]
        batch_rgbs = [f.rgb for f in batch]
        try:
            batch_dets = localizer.detect_batch(batch_rgbs, detect_query)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [{_ts()}] OOM, falling back to single-frame")
                torch.cuda.empty_cache()
                batch_dets = []
                for f in batch:
                    try:
                        dets = localizer.detect(f.rgb, detect_query)
                        batch_dets.append(dets)
                    except RuntimeError:
                        batch_dets.append([])
            else:
                raise

        for frame, dets in zip(batch, batch_dets):
            for d in dets:
                q = score_detection(d, frame.rgb)
                candidates.append({
                    "frame_index": frame.index,
                    "timestamp_sec": frame.timestamp_sec,
                    "bbox": d.bbox,
                    "confidence": d.score,
                    "quality_score": q,
                    "sharpness": laplacian_sharpness(frame.rgb),
                    "frame_sample": frame,
                })
        pct = min(100, int(100 * (bi + len(batch)) / len(candidate_frames)))
        print(f"  [{_ts()}] {pct:3d}% | detections={len(candidates)}")

    elapsed = time.time() - t0
    del localizer
    gc.collect()
    torch.cuda.empty_cache()
    return candidates, elapsed


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    start = time.time()
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{_ts()}] Pipeline started")
    print(f"  video: {cfg.video_path}")
    print(f"  query: {cfg.query}")
    print(f"  fps={cfg.fps}, clip_topk={cfg.clip_topk}, max_side={cfg.max_side}")

    # ── Stage 1: LLM ────────────────────────────────────────────
    print(f"\n[{_ts()}] [1/5] LLM target extraction ...")
    t0 = time.time()
    spec = parse_target_with_llm(
        cfg.query,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
    )
    clip_query = spec.clip_query
    detect_query = spec.detect_query
    print(f"  -> CLIP query:    {clip_query!r}")
    print(f"  -> OWLv2 query:   {detect_query!r}  ({time.time()-t0:.1f}s)")

    # ── Stage 2: Sampling ───────────────────────────────────────
    print(f"\n[{_ts()}] [2/5] Sampling video frames ...")
    t0 = time.time()
    frames, sample_meta = sample_frames(
        cfg.video_path,
        fps=cfg.fps,
        sampling_mode=cfg.sampling_mode,
        max_side=cfg.max_side,
    )
    if not frames:
        print(f"[{_ts()}] [ERROR] No frames could be sampled")
        return {"error": "no_frames", "elapsed_sec": time.time() - start}
    print(f"[{_ts()}] {len(frames)} frames sampled ({time.time()-t0:.1f}s)")

    # ── Stage 3: CLIP ───────────────────────────────────────────
    print(f"\n[{_ts()}] [3/5] CLIP coarse retrieval on {len(frames)} frames ...")
    t0 = time.time()
    rgbs_all = [f.rgb for f in frames]
    clip_model = CLIPRetriever(model_name="ViT-B/32")
    text_vec = clip_model.encode_text(clip_query)
    clip_scores = clip_model.score_frames(rgbs_all, text_vec, batch_size=cfg.clip_batch_size)

    scored = sorted(zip(frames, clip_scores), key=lambda x: x[1], reverse=True)
    topk = min(cfg.clip_topk, len(frames))
    candidate_frames = [f for f, _ in scored[:topk]]
    top_scores = [round(s, 4) for _, s in scored[:topk]]
    print(f"[{_ts()}] CLIP top-{topk} scores: {top_scores[:5]}... ({time.time()-t0:.1f}s)")
    del clip_model, text_vec, rgbs_all, scored
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 4: OWLv2 with threshold retry ─────────────────────
    thresholds_to_try = [cfg.box_threshold, 0.15, 0.10, 0.05]
    candidates: List[Dict[str, Any]] = []
    owl_elapsed = 0.0

    for thresh in thresholds_to_try:
        print(f"\n[{_ts()}] [4/5] OWLv2 grounding (threshold={thresh}) ...")
        candidates, owl_elapsed = _run_owlv2_stage(
            candidate_frames, detect_query,
            box_threshold=thresh,
            text_threshold=cfg.text_threshold,
            owl_batch_size=cfg.owl_batch_size,
        )
        print(f"[{_ts()}] OWLv2: {len(candidates)} detections ({owl_elapsed:.1f}s)")
        if candidates:
            break

    # ── Stage 5: Output ─────────────────────────────────────────
    if not candidates:
        elapsed = time.time() - start
        print(f"\n[{_ts()}] [5/5] No target detected after all thresholds")
        print(f"  CLIP top scores: {top_scores[:10]}")
        print(f"  CLIP query:    {clip_query!r}")
        print(f"  OWLv2 query:   {detect_query!r}")
        print(f"  Tried thresholds: {thresholds_to_try}")
        print(f"  === EXIT (no result) === ({elapsed:.1f}s)")
        return {
            "video": cfg.video_path,
            "query": cfg.query,
            "clip_query": clip_query,
            "detect_query": detect_query,
            "error": "no_detection",
            "clip_top_scores": top_scores,
            "tried_thresholds": thresholds_to_try,
            "sampling": sample_meta,
            "elapsed_sec": elapsed,
        }

    print(f"\n[{_ts()}] [5/5] Selecting best target crop ...")
    best = max(candidates, key=lambda c: c["quality_score"])
    best_frame: FrameSample = best["frame_sample"]
    crop_rgb = crop_with_padding(best_frame.rgb, best["bbox"], cfg.crop_padding_ratio)

    elapsed = time.time() - start
    meta = {
        "video": cfg.video_path,
        "query": cfg.query,
        "clip_query": clip_query,
        "detect_query": detect_query,
        "crop_prompt": spec.crop_prompt,
        "best_timestamp_sec": best["timestamp_sec"],
        "bbox": best["bbox"],
        "confidence": best["confidence"],
        "quality_score": best["quality_score"],
        "sharpness": best["sharpness"],
        "clip_top_scores": top_scores,
        "sampling": sample_meta,
        "candidate_count": len(candidates),
        "models": {
            "llm": cfg.llm_model or "deepseek-v4-flash",
            "coarse_retriever": "CLIP-ViT-B/32",
            "localizer": "google/owlv2-base-patch16-ensemble",
            "scorer": "heuristic",
        },
        "elapsed_sec": elapsed,
    }

    out_dir = out_root / Path(cfg.video_path).stem
    paths = save_outputs(out_dir, best_frame.rgb, crop_rgb, meta)

    print(f"\n[{_ts()}] ===== DONE ({elapsed:.1f}s) =====")
    print(f"  detect_query : {detect_query}")
    print(f"  clip_query   : {clip_query}")
    print(f"  timestamp    : {meta['best_timestamp_sec']:.2f}s")
    print(f"  confidence   : {meta['confidence']:.3f}")
    print(f"  quality      : {meta['quality_score']:.3f}")
    print(f"  crop         : {paths['target_crop']}")
    print(f"  meta         : {paths['meta_json']}")
    return meta

from __future__ import annotations

import json
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_target_extractor import parse_target_with_llm
from .video_sampler import sample_frames, FrameSample
from .clip_retriever import CLIPRetriever
from .text_localizer import OwlV2Localizer, OWLConfig
from .frame_scorer import score_detection, laplacian_sharpness
from .cropper import crop_with_padding, save_outputs
from .detection import Detection


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


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    start = time.time()
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Stage 0: LLM target extraction ──────────────────────────
    print("[1/5] LLM target extraction ...")
    t0 = time.time()
    spec = parse_target_with_llm(
        cfg.query,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
    )
    query_for_det = spec.english_query or spec.target_object or cfg.query
    print(f"  -> target={spec.target_object!r}, english_query={query_for_det!r} ({time.time()-t0:.2f}s)")

    # ── Stage 1: Video frame sampling ───────────────────────────
    print("[2/5] Sampling video frames ...")
    t0 = time.time()
    frames, sample_meta = sample_frames(
        cfg.video_path,
        fps=cfg.fps,
        sampling_mode=cfg.sampling_mode,
        max_side=cfg.max_side,
    )
    if not frames:
        raise RuntimeError("No frames sampled from video")
    print(f"  -> {len(frames)} frames sampled in {time.time()-t0:.2f}s")

    # ── Stage 2: CLIP coarse retrieval ──────────────────────────
    print("[3/5] CLIP coarse retrieval ...")
    t0 = time.time()
    rgbs_all = [f.rgb for f in frames]
    clip_model = CLIPRetriever(model_name="ViT-B/32")
    text_vec = clip_model.encode_text(query_for_det)
    clip_scores = clip_model.score_frames(rgbs_all, text_vec, batch_size=cfg.clip_batch_size)

    scored = sorted(
        zip(frames, clip_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    topk = min(cfg.clip_topk, len(frames))
    candidate_frames = [f for f, _ in scored[:topk]]
    top_scores = [s for _, s in scored[:topk]]
    del clip_model, text_vec, rgbs_all, scored  # free VRAM
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  -> CLIP scored {len(frames)} frames, top-{topk} selected ({time.time()-t0:.2f}s)")

    # ── Stage 3: OWLv2 fine-grained detection ───────────────────
    print(f"[4/5] OWLv2 grounding on top-{topk} frames ...")
    t0 = time.time()
    localizer = OwlV2Localizer(
        OWLConfig(box_threshold=cfg.box_threshold, text_threshold=cfg.text_threshold)
    )

    candidates: List[Dict[str, Any]] = []
    # Process in batches to save VRAM
    bs = cfg.owl_batch_size if cfg.owl_batch_size >= 1 else 1
    for bi in range(0, len(candidate_frames), bs):
        batch = candidate_frames[bi : bi + bs]
        batch_rgbs = [f.rgb for f in batch]
        try:
            batch_dets = localizer.detect_batch(batch_rgbs, query_for_det)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM on batch, falling back to single-frame mode")
                torch.cuda.empty_cache()
                batch_dets = []
                for f in batch:
                    try:
                        dets = localizer.detect(f.rgb, query_for_det)
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
        if (bi // bs) % 5 == 0:
            pct = min(100, int(100 * (bi + len(batch)) / len(candidate_frames)))
            print(f"  grounded {pct}% frames, detections so far={len(candidates)}")

    del localizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  -> {len(candidates)} detections across {len(candidate_frames)} frames ({time.time()-t0:.2f}s)")

    if not candidates:
        print("[WARN] No target object detected in candidate frames")
        print("  CLIP top scores:", top_scores[:10])
        raise RuntimeError(
            f"No target object detected. CLIP found top-{topk} frames but OWLv2 found no bbox. "
            f"Try lowering --box_threshold (current: {cfg.box_threshold}) or checking the query."
        )

    # ── Stage 4: Select best target crop ────────────────────────
    print("[5/5] Selecting best target crop ...")
    best = max(candidates, key=lambda c: c["quality_score"])
    best_frame: FrameSample = best["frame_sample"]
    crop_rgb = crop_with_padding(best_frame.rgb, best["bbox"], cfg.crop_padding_ratio)

    meta = {
        "video": cfg.video_path,
        "query": cfg.query,
        "target_object": spec.target_object,
        "english_query": query_for_det,
        "best_timestamp_sec": best["timestamp_sec"],
        "bbox": best["bbox"],
        "confidence": best["confidence"],
        "quality_score": best["quality_score"],
        "sharpness": best["sharpness"],
        "clip_top_scores": top_scores[:10],
        "sampling": sample_meta,
        "candidate_count": len(candidates),
        "models": {
            "llm": cfg.llm_model or "deepseek-v4-flash",
            "coarse_retriever": "CLIP-ViT-B/32",
            "localizer": "google/owlv2-base-patch16-ensemble",
            "scorer": "heuristic",
        },
        "elapsed_sec": time.time() - start,
    }

    out_dir = out_root / Path(cfg.video_path).stem
    paths = save_outputs(out_dir, best_frame.rgb, crop_rgb, meta)

    print(f"\n=== Done ({meta['elapsed_sec']:.1f}s total) ===")
    print(f"  target: {spec.target_object!r}")
    print(f"  best_timestamp: {meta['best_timestamp_sec']:.3f}s")
    print(f"  confidence: {meta['confidence']:.3f}")
    print(f"  quality_score: {meta['quality_score']:.3f}")
    print(f"  outputs: {json.dumps(paths, ensure_ascii=False)}")
    return meta

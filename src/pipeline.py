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
from .detectors import create_detector, BaseDetector
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
    detector: str = "owlv2"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    crop_padding_ratio: float = 0.08
    det_batch_size: int = 4
    gdin_batch_size: int = 4


def _run_detector_stage(
    frames: List[FrameSample],
    detect_query: str,
    detector: BaseDetector,
    box_threshold: float,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = time.time()
    # Set threshold on detector
    detector.box_threshold = box_threshold

    candidates: List[Dict[str, Any]] = []
    bs = max(1, batch_size)
    total = len(frames)

    for bi in range(0, total, bs):
        batch = frames[bi : bi + bs]
        batch_rgbs = [f.rgb for f in batch]
        try:
            batch_dets = detector.detect_batch(batch_rgbs, detect_query)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [{_ts()}] OOM, falling back to single-frame")
                torch.cuda.empty_cache()
                batch_dets = []
                for f in batch:
                    try:
                        dets = detector.detect(f.rgb, detect_query)
                        batch_dets.append(dets)
                    except RuntimeError:
                        batch_dets.append([])
            else:
                raise

        for frame, dets in zip(batch, batch_dets):
            for d in dets:
                candidates.append({
                    "frame_index": frame.index,
                    "timestamp_sec": frame.timestamp_sec,
                    "bbox": d.bbox,
                    "confidence": d.score,
                    "quality_score": d.score,
                    "frame_sample": frame,
                })
        pct = min(100, int(100 * (bi + len(batch)) / total))
        print(f"  [{_ts()}] {pct:3d}% | detections={len(candidates)}")

    elapsed = time.time() - t0
    return candidates, elapsed


def _print_timing(timing: dict, total_frames: int, detector_name: str):
    print("")
    print(f"  {'='*55}")
    print(f"  Timing Summary  (detector: {detector_name})")
    print(f"  {'='*55}")
    print(f"  Stage 1 (LLM parsing)  : {timing.get('1_llm', 0):>8.2f}s")
    print(f"  Stage 2 (Sampling)    : {timing.get('2_sampling', 0):>8.2f}s  ({total_frames} frames)")
    dt = timing.get('3_detection', 0)
    print(f"  Stage 3 (Detection)   : {dt:>8.2f}s")
    if '3a_model_load' in timing:
        print(f"    - model load        : {timing['3a_model_load']:>8.3f}s")
    print(f"  Stage 4 (Crop+Save)   : {timing.get('4_output', 0):>8.2f}s")
    print(f"  {'-'*55}")
    print(f"  TOTAL                 : {timing.get('total', 0):>8.2f}s")
    print(f"  {'='*55}")


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    timing = {}
    start = time.time()
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{_ts()}] Pipeline started")
    print(f"  video: {cfg.video_path}")
    print(f"  query: {cfg.query}")
    print(f"  fps={cfg.fps}, detector={cfg.detector}, max_side={cfg.max_side}")

    # Stage 1: LLM
    print(f"\n[{_ts()}] [1/4] LLM target extraction ...")
    t1 = time.time()
    spec = parse_target_with_llm(
        cfg.query,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
    )
    detect_query = spec.detect_query
    t_llm = round(time.time() - t1, 2)
    timing['1_llm'] = t_llm
    print(f"  -> detect query: {detect_query!r}  ({t_llm}s)")

    # Stage 2: Sampling
    print(f"\n[{_ts()}] [2/4] Sampling video frames ...")
    t2 = time.time()
    frames, sample_meta = sample_frames(
        cfg.video_path,
        fps=cfg.fps,
        sampling_mode=cfg.sampling_mode,
        max_side=cfg.max_side,
    )
    t_sampling = round(time.time() - t2, 2)
    timing['2_sampling'] = t_sampling

    if not frames:
        print(f"[{_ts()}] [ERROR] No frames could be sampled")
        return {"error": "no_frames", "elapsed_sec": time.time() - start}
    n_frames = len(frames)
    print(f"[{_ts()}] {n_frames} frames sampled ({t_sampling}s)")

    # Stage 3: Detection (all frames, with threshold retry)
    print(f"\n[{_ts()}] [3/4] Detection on {n_frames} frames (detector={cfg.detector}) ...")

    # Determine batch size based on detector
    if cfg.detector.lower() == "grounding_dino":
        det_bs = cfg.gdin_batch_size
    else:
        det_bs = cfg.det_batch_size

    ta = time.time()
    detector = create_detector(cfg.detector, box_threshold=cfg.box_threshold)
    t3a = round(time.time() - ta, 3)
    timing['3a_model_load'] = t3a

    thresholds_to_try = [cfg.box_threshold, 0.15, 0.10, 0.05]
    candidates: List[Dict[str, Any]] = []
    det_elapsed = 0.0

    for thresh in thresholds_to_try:
        # Only recreate detector if threshold changed (first iteration uses existing)
        if thresh != thresholds_to_try[0]:
            print(f"\n  [{_ts()}] Retrying with threshold={thresh} ...")
        candidates, stage_elapsed = _run_detector_stage(
            frames, detect_query, detector,
            box_threshold=thresh,
            batch_size=det_bs,
        )
        det_elapsed += stage_elapsed
        print(f"[{_ts()}] {cfg.detector}: {len(candidates)} detections ({stage_elapsed:.1f}s)")
        if candidates:
            break

    t_detection = round(det_elapsed, 2)
    timing['3_detection'] = t_detection

    del detector
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 4: Output
    if not candidates:
        timing['4_output'] = 0.0
        t_total = round(time.time() - start, 2)
        timing['total'] = t_total
        _print_timing(timing, n_frames, cfg.detector)
        print(f"\n[{_ts()}] [4/4] No target detected after all thresholds")
        print(f"  detect query:  {detect_query!r}")
        print(f"  Tried thresholds: {thresholds_to_try}")
        print(f"  Detector: {cfg.detector}")
        print(f"  === EXIT (no result) === ({t_total}s)")
        return {
            "video": cfg.video_path,
            "query": cfg.query,
            "detect_query": detect_query,
            "error": "no_detection",
            "tried_thresholds": thresholds_to_try,
            "detector": cfg.detector,
            "sampling": sample_meta,
            "elapsed_sec": t_total,
            "timing": timing,
        }

    t4 = time.time()
    print(f"\n[{_ts()}] [4/4] Selecting best target crop ...")
    best = max(candidates, key=lambda c: c["quality_score"])
    best_frame: FrameSample = best["frame_sample"]
    crop_rgb = crop_with_padding(best_frame.rgb, best["bbox"], cfg.crop_padding_ratio)
    t_output = round(time.time() - t4, 2)
    timing['4_output'] = t_output

    t_total = round(time.time() - start, 2)
    timing['total'] = t_total
    _print_timing(timing, n_frames, cfg.detector)

    meta = {
        "video": cfg.video_path,
        "query": cfg.query,
        "detect_query": detect_query,
        "crop_prompt": spec.crop_prompt,
        "best_timestamp_sec": best["timestamp_sec"],
        "bbox": best["bbox"],
        "confidence": best["confidence"],
        "quality_score": best["quality_score"],
        "sampling": sample_meta,
        "candidate_count": len(candidates),
        "models": {
            "llm": cfg.llm_model or "deepseek-v4-flash",
            "detector": cfg.detector,
            "scorer": "confidence_only",
        },
        "elapsed_sec": t_total,
        "timing": timing,
    }

    out_dir = out_root / Path(cfg.video_path).stem
    paths = save_outputs(out_dir, best_frame.rgb, crop_rgb, meta)

    print(f"\n[{_ts()}] ===== DONE ({t_total}s) =====")
    print(f"  detect_query : {detect_query}")
    print(f"  detector     : {cfg.detector}")
    print(f"  timestamp    : {meta['best_timestamp_sec']:.2f}s")
    print(f"  confidence   : {meta['confidence']:.3f}")
    print(f"  crop         : {paths['target_crop']}")
    print(f"  meta         : {paths['meta_json']}")
    return meta

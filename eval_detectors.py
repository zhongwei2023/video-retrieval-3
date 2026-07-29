#!/usr/bin/env python
"""Evaluate OWLv2 vs Grounding DINO on Refer-YouTube-VOS.

Quick start (30 videos):
    python eval_detectors.py --max-videos 30

Full run (507 videos):
    python eval_detectors.py --max-videos 507 --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.detectors import create_detector, create_detector_with_text_threshold
from src.detection import Detection


# ---- prompt normalization ----

def normalize_expression_to_query(expression: str) -> str:
    """Convert a verbose referring expression to a short detection-friendly prompt.

    Grounding models (and even OWL) perform better on compact noun phrases
    than on long, positional sentences.
    """
    t = expression.strip().rstrip(".")
    # drop leading articles
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.I).strip()
    # cut at common verb phrase starts
    m = re.search(r"\b(is|are|was|were|sits|sitting|standing|walking|carrying|showing|drinking|goes|goes in to)\b", t, flags=re.I)
    if m and m.start() > 0:
        t = t[:m.start()].strip()
    # drop trailing prepositional phrases
    t = re.sub(r"\b(in|on|at|from|with|of|near|behind|towards|another|around)\b.*$", "", t, flags=re.I).strip()
    if not t:
        t = " ".join(expression.strip().rstrip(".").split()[:3])
    return t.strip().rstrip(".") + " ."


# ---- data loading ----

def load_meta(dataset_root: Path) -> Tuple[Dict, Dict]:
    meta_path = dataset_root / "meta.json"
    expr_path = dataset_root / "meta_expressions" / "meta_expressions" / "valid" / "meta_expressions.json"
    with open(meta_path) as f:
        meta = json.load(f)
    with open(expr_path) as f:
        expressions = json.load(f)
    return meta, expressions


def get_gt_bbox(anno_dir: Path, video_id: str, object_id: int, frame_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract bbox [x1,y1,x2,y2] from annotation PNG for a given object."""
    anno_path = anno_dir / video_id / f"{frame_name}.png"
    if not anno_path.exists():
        return None
    mask = np.array(Image.open(anno_path))
    ys, xs = np.where(mask == object_id)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def compute_iou(box1: List[float], box2: Tuple[float, ...]) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 1e-8 else 0.0


# ---- evaluation ----

def build_samples(meta: Dict, expressions: Dict, max_videos: int) -> List[Dict]:
    """Build list of (video, object, frame, expression) samples to evaluate."""
    samples = []
    video_ids = sorted(expressions["videos"].keys())

    for vid in video_ids[:max_videos]:
        if vid not in meta["videos"]:
            continue

        obj_data = meta["videos"][vid]["objects"]
        expr_video = expressions["videos"][vid]
        expr_dict = expr_video["expressions"]

        n_objects = len(obj_data)
        n_exprs = len(expr_dict)
        if n_objects == 0 or n_exprs == 0:
            continue
        n_per_obj = n_exprs // n_objects

        # All objects share the same annotated frame list -- pick the first frame
        first_obj_frames = list(obj_data.values())[0]["frames"]
        if not first_obj_frames:
            continue
        eval_frame = first_obj_frames[0]

        for expr_idx_str, expr_info in expr_dict.items():
            expr_idx = int(expr_idx_str)
            obj_idx = expr_idx // n_per_obj
            if obj_idx >= n_objects:
                continue
            obj_id_str = list(obj_data.keys())[obj_idx]

            samples.append({
                "video_id": vid,
                "object_id": int(obj_id_str),
                "frame_name": eval_frame,
                "expression": expr_info["exp"],
            })

    return samples


def run_one(
    detector,
    frame_path: Path,
    expression: str,
    gt_bbox: Tuple[float, float, float, float],
    name: str,
) -> Tuple[float, float, Optional[List[float]]]:
    """Run detector on one frame, return (iou, confidence, bbox)."""
    try:
        rgb = np.array(Image.open(frame_path).convert("RGB"))
        dets = detector.detect(rgb, expression)
        if dets:
            best = max(dets, key=lambda d: d.score)
            return compute_iou(best.bbox, gt_bbox), best.score, best.bbox
        return 0.0, 0.0, None
    except Exception as e:
        print(f"  [{name}] error: {e}")
        return 0.0, 0.0, None


# ---- main ----

def main():
    parser = argparse.ArgumentParser(description="Compare OWLv2 vs Grounding DINO on Refer-YouTube-VOS")
    parser.add_argument("--dataset-root",
                        default=r"F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid")
    parser.add_argument("--max-videos", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--box-threshold", type=float, default=0.10,
                        help="Lower threshold to capture more detections for fair comparison")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    jpeg_dir = dataset_root / "JPEGImages"
    anno_dir = dataset_root / "Annotations"

    print("Loading annotations...")
    meta, expressions = load_meta(dataset_root)
    samples = build_samples(meta, expressions, args.max_videos)
    n_videos = min(args.max_videos, len(expressions["videos"]))
    print(f"  {len(samples)} evaluation samples from {n_videos} videos")

    # ---- load detectors ----
    print("\nLoading OWLv2 ...")
    t0 = time.time()
    det_owl = create_detector("owlv2", box_threshold=args.box_threshold, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Loading Grounding DINO ...")
    t0 = time.time()
    det_dino = create_detector_with_text_threshold(
        "grounding_dino",
        box_threshold=args.box_threshold,
        text_threshold=max(0.05, args.box_threshold * 0.5),
        device=args.device,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ---- evaluate ----
    results = []
    start = time.time()

    for i, s in enumerate(samples):
        vid, obj_id, frame_name, expr = s["video_id"], s["object_id"], s["frame_name"], s["expression"]

        frame_path = jpeg_dir / vid / f"{frame_name}.jpg"
        if not frame_path.exists():
            continue

        gt_bbox = get_gt_bbox(anno_dir, vid, obj_id, frame_name)
        if gt_bbox is None:
            continue

        owl_query = normalize_expression_to_query(expr)
        owl_iou, owl_conf, owl_bbox = run_one(det_owl, frame_path, owl_query, gt_bbox, "OWLv2")

        dino_query = normalize_expression_to_query(expr)
        dino_iou, dino_conf, dino_bbox = run_one(det_dino, frame_path, dino_query, gt_bbox, "DINO")

        results.append({
            "video": vid,
            "object": obj_id,
            "frame": frame_name,
            "expression": expr,
            "owl_query": owl_query,
            "dino_query": dino_query,
            "owl_iou": owl_iou,
            "owl_conf": owl_conf,
            "owl_bbox": owl_bbox,
            "dino_iou": dino_iou,
            "dino_conf": dino_conf,
            "dino_bbox": dino_bbox,
            "gt_bbox": list(gt_bbox),
        })

        if (i + 1) % 30 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s  (last OWL IoU={owl_iou:.3f}, DINO IoU={dino_iou:.3f})")

    # ---- summary ----
    valid = [r for r in results if "owl_iou" in r]
    if not valid:
        print("No valid results!")
        return

    owl_ious = np.array([r["owl_iou"] for r in valid])
    dino_ious = np.array([r["dino_iou"] for r in valid])
    owl_confs = np.array([r["owl_conf"] for r in valid])
    dino_confs = np.array([r["dino_conf"] for r in valid])

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Videos: {n_videos}   Samples: {len(valid)}")
    print(f"  {'':25s} {'OWLv2':>15s} {'Grounding DINO':>16s}")
    print(f"  {'-'*57}")
    print(f"  {'Mean IoU':25s} {np.mean(owl_ious):15.4f} {np.mean(dino_ious):16.4f}")
    print(f"  {'Median IoU':25s} {np.median(owl_ious):15.4f} {np.median(dino_ious):16.4f}")
    print(f"  {'Recall @ IoU >= 0.5':25s} {np.mean(owl_ious >= 0.5)*100:14.1f}% {np.mean(dino_ious >= 0.5)*100:15.1f}%")
    print(f"  {'Recall @ IoU >= 0.3':25s} {np.mean(owl_ious >= 0.3)*100:14.1f}% {np.mean(dino_ious >= 0.3)*100:15.1f}%")
    print(f"  {'Avg Confidence':25s} {np.mean(owl_confs):15.4f} {np.mean(dino_confs):16.4f}")
    print(f"  {'Detections (>0 IoU)':25s} {np.sum(owl_ious > 0):15d} {np.sum(dino_ious > 0):16d}")
    print("=" * 65)

    # IoU delta
    deltas = owl_ious - dino_ious
    print(f"\n  IoU Delta (OWL - DINO): mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}")
    print(f"  OWL wins:  {np.sum(deltas > 0.01):3d} samples")
    print(f"  DINO wins: {np.sum(deltas < -0.01):3d} samples")
    print(f"  Tie:       {np.sum(np.abs(deltas) <= 0.01):3d} samples")

    # ---- per-sample top and bottom ----
    print(f"\n  Top-5 OWL wins:")
    idx_sorted = np.argsort(deltas)[::-1]
    for idx in idx_sorted[:5]:
        r = valid[idx]
        print(f"    {r['video']} obj{r['object']}: OWL={deltas[idx]+valid[idx]['dino_iou']:.3f} DINO={valid[idx]['dino_iou']:.3f}  | {r['expression'][:60]}")

    print(f"\n  Top-5 DINO wins:")
    for idx in idx_sorted[-5:][::-1]:
        r = valid[idx]
        print(f"    {r['video']} obj{r['object']}: OWL={deltas[idx]+valid[idx]['dino_iou']:.3f} DINO={valid[idx]['dino_iou']:.3f}  | {r['expression'][:60]}")

    # ---- save ----
    out_path = dataset_root.parent / "eval_results.json"
    summary = {
        "n_videos": n_videos,
        "n_samples": len(valid),
        "owl_mean_iou": float(np.mean(owl_ious)),
        "dino_mean_iou": float(np.mean(dino_ious)),
        "owl_recall_05": float(np.mean(owl_ious >= 0.5)),
        "dino_recall_05": float(np.mean(dino_ious >= 0.5)),
        "owl_avg_conf": float(np.mean(owl_confs)),
        "dino_avg_conf": float(np.mean(dino_confs)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": valid}, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
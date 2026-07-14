from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video target frame extraction MVP")
    parser.add_argument("--video", required=True, help="Local mp4 video path")
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="Sampling fps override")
    parser.add_argument("--sampling_mode", default="high_density", choices=["high_density", "all_frames"], help="Sampling mode if fps is not set")
    parser.add_argument("--max_side", type=int, default=1024, help="Resize long side before detection")
    parser.add_argument("--llm_base_url", default=None, help="LLM base url")
    parser.add_argument("--llm_model", default=None, help="LLM model name")
    parser.add_argument("--llm_api_key", default=None, help="LLM API key")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="OWLv2 box threshold")
    parser.add_argument("--clip_topk", type=int, default=30, help="Top-K frames from CLIP to send to OWLv2")
    parser.add_argument("--clip_batch", type=int, default=32, help="CLIP inference batch size")
    parser.add_argument("--owl_batch", type=int, default=1, help="OWLv2 inference batch size (1=safer for 4GB)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        video_path=args.video,
        query=args.query,
        out_dir=args.out_dir,
        fps=args.fps,
        sampling_mode=args.sampling_mode,
        max_side=args.max_side,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        box_threshold=args.box_threshold,
        clip_topk=args.clip_topk,
        clip_batch_size=args.clip_batch,
        owl_batch_size=args.owl_batch,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

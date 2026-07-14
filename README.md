# Video Target Frame Extraction MVP

Two-stage pipeline: CLIP coarse retrieval -> OWLv2 fine detection

## Pipeline

```
Video -> Sample frames -> CLIP scores all -> Top-K frames -> OWLv2 detects bbox -> Pick clearest -> Crop output
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.cli --video "F:/path/to/video.mp4" --query "video question" --out_dir "F:/output" --llm_api_key "YOUR_KEY"
```

## Key Args

| arg | default | description |
|-----|---------|-------------|
| --video | required | local mp4 path |
| --query | required | natural language question |
| --out_dir | required | output directory |
| --fps | 2-4 (auto) | sampling fps, higher = less likely to miss frames |
| --sampling_mode | high_density | high_density or all_frames |
| --max_side | 1024 | resize long side before detection |
| --clip_topk | 30 | top-K CLIP frames to run OWLv2 on |
| --clip_batch | 32 | CLIP inference batch size |
| --owl_batch | 1 | OWLv2 batch size (1=safer for 4GB, 4-8 for 4090) |
| --box_threshold | 0.25 | OWLv2 detection threshold |
| --llm_api_key | - | DeepSeek API key |

## 3050 Ti recommended

```bash
python -m src.cli --video "F:/video.mp4" --query "question" --out_dir "F:/output" --fps 3 --clip_topk 30 --owl_batch 1 --max_side 960 --llm_api_key "KEY"
```

## 4090 recommended

```bash
python -m src.cli --video "F:/video.mp4" --query "question" --out_dir "F:/output" --fps 5 --clip_topk 50 --owl_batch 4 --max_side 1536 --llm_api_key "KEY"
```

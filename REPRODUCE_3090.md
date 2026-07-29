# 3090 复现与评估指南

本文用于在 NVIDIA RTX 3090 24GB 机器上复现视频目标定位流程，并使用 Refer-YouTube-VOS 验证 OWLv2 与 Grounding DINO 的检测效果。

## 1. 环境准备

进入项目目录并创建环境：

```powershell
cd F:\project_of_codex\video-retrieval_3

conda create -n video_retrieval python=3.10 -y
conda activate video_retrieval
pip install -r requirements.txt
```

建议确认 CUDA 和 PyTorch 可用：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

预期 `torch.cuda.is_available()` 为 `True`，显卡名称为 RTX 3090。若出现 Windows `WinError 1455`，通常是系统页面文件过小，需要增大页面文件后重启机器。

## 2. 数据目录检查

当前数据集根目录应为：

```text
F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid
```

目录结构：

```text
valid/
├── JPEGImages/
│   ├── <video_id>/00000.jpg
│   └── ...
├── Annotations/
│   ├── <video_id>/00000.png
│   └── ...
├── meta.json
└── meta_expressions/
    └── meta_expressions/
        └── valid/
            └── meta_expressions.json
```

当前检查结果应为：

- 507 个视频目录；
- `JPEGImages` 与 `Annotations` 的视频 ID 对齐；
- `meta.json` 与 `meta_expressions.json` 包含同一批视频；
- 标注 PNG 是 palette 图像，像素索引 `1、2、3...` 对应 `meta.json` 中的 object ID；
- YouTube-VOS 验证集是稀疏标注，通常每 5 帧提供一张标注图。

检查文件是否存在：

```powershell
Test-Path "F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid\meta.json"
Test-Path "F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid\meta_expressions\meta_expressions\valid\meta_expressions.json"
```

## 3. 单视频流程复现

当前流程已经移除 CLIP 粗筛，抽样后的全部帧都会进入检测器。3090 上先用 batch size 4；若显存不足，将 batch 降为 2 或 1。

### OWLv2

```powershell
python -m src.cli `
  --video "F:\project_of_codex\video-retrieval_3\video\test_6.mp4" `
  --query "What color are the shoes of the child wearing the fluorescent yellow clothes?" `
  --fps 15 `
  --max_side 1024 `
  --out_dir "F:\project_of_codex\video-retrieval_3\output\owlv2" `
  --llm_model "deepseek-v4-flash" `
  --llm_api_key "YOUR_DEEPSEEK_API_KEY" `
  --detector owlv2 `
  --det_batch 4
```

### Grounding DINO

```powershell
python -m src.cli `
  --video "F:\project_of_codex\video-retrieval_3\video\test_6.mp4" `
  --query "What color are the shoes of the child wearing the fluorescent yellow clothes?" `
  --fps 15 `
  --max_side 1024 `
  --out_dir "F:\project_of_codex\video-retrieval_3\output\grounding_dino" `
  --llm_model "deepseek-v4-flash" `
  --llm_api_key "YOUR_DEEPSEEK_API_KEY" `
  --detector grounding_dino `
  --gdin_batch 4
```

首次运行会从 Hugging Face 下载模型：

- OWLv2：`google/owlv2-base-patch16-ensemble`；
- Grounding DINO：`IDEA-Research/grounding-dino-base`。

模型下载完成后会缓存到 Hugging Face 缓存目录，后续运行不需要重复下载。若服务器无法访问 Hugging Face，应先在可联网机器下载模型，再设置 `HF_HOME` 或复制 Hugging Face 缓存。

每次成功运行会在对应输出目录下生成：

```text
<out_dir>/test_6/
├── best_frame.jpg
├── target_crop.jpg
└── meta.json
```

`meta.json` 中重点查看：

- `best_timestamp_sec`：最终选中的时间点；
- `bbox`：检测框；
- `confidence`：检测置信度；
- `quality_score`：当前等于 `confidence`；
- `candidate_count`：全部候选检测数量；
- `timing`：LLM、抽帧、检测和裁切耗时。

## 4. 先做小规模评估

评估脚本位于：`eval_detectors.py`。

先跑 5 个视频确认模型、数据和显存配置：

```powershell
python eval_detectors.py --max-videos 5 --device cuda --box-threshold 0.10
```

然后跑 30 个视频作为快速实验：

```powershell
python eval_detectors.py `
  --max-videos 30 `
  --device cuda `
  --box-threshold 0.10
```

脚本默认使用项目中的数据路径。如果数据放在其他位置，显式指定：

```powershell
python eval_detectors.py `
  --dataset-root "F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid" `
  --max-videos 30 `
  --device cuda `
  --box-threshold 0.10
```

## 5. 完整评估

确认 5 或 30 个视频能够正常完成后，再运行全部 507 个视频：

```powershell
python eval_detectors.py `
  --dataset-root "F:\project_of_codex\video-retrieval_3\dataset-yt\valid\valid" `
  --max-videos 507 `
  --device cuda `
  --box-threshold 0.10
```

脚本会为每条指代表达选取其第一个有标注的帧，在该帧上分别运行两个检测器，并计算：

- Mean IoU；
- Median IoU；
- Recall@IoU 0.5；
- Recall@IoU 0.3；
- 平均 confidence；
- 两个检测器的胜负次数。

结果保存到：

```text
F:\project_of_codex\video-retrieval_3\dataset-yt\valid\eval_results.json
```

## 6. 结果如何用于消融实验

重点报告以下表格：

| 指标 | OWLv2 | Grounding DINO |
|---|---:|---:|
| Mean IoU | 运行后填写 | 运行后填写 |
| Median IoU | 运行后填写 | 运行后填写 |
| Recall@0.5 | 运行后填写 | 运行后填写 |
| Recall@0.3 | 运行后填写 | 运行后填写 |
| Average confidence | 运行后填写 | 运行后填写 |

如果两个检测器的定位指标接近，说明后续系统的提升不能简单归因于某一个检测器。建议同时报告绝对差值，例如：

```text
Delta Mean IoU = Mean IoU(ours detector) - Mean IoU(baseline detector)
```

需要注意：该评估脚本验证的是“单帧指代表达检测能力”，不能完全代表完整视频流程的帧选择能力。若要验证帧选择，还需要在每个视频的全部稀疏标注帧上运行 pipeline，并比较最终选中帧与目标清晰度或人工标注结果。

## 7. 显存和运行建议

3090 24GB 的推荐起点：

```text
max_side       = 1024
det_batch      = 4
gdin_batch     = 4
box_threshold  = 0.10（评估时）或 0.25（默认推理时）
device         = cuda
```

遇到 CUDA out of memory 时，按以下顺序调整：

1. 将 `--det_batch` 或 `--gdin_batch` 降到 2；
2. 再降到 1；
3. 将 `--max_side` 降到 768 或 640；
4. 检查是否同时运行了其他占用显存的程序。

每次正式对比都应保持以下设置一致：视频、query、抽样 FPS、`max_side`、检测阈值和输出评价规则。只改变 `--detector`，这样比较才属于有效消融。

## 8. 常见问题

### 找不到标注文件

确认 `--dataset-root` 指向包含 `meta.json`、`JPEGImages` 和 `Annotations` 的 `valid` 目录，而不是上一级目录。

### 只能使用 CPU

可以用下面的命令测试流程，但不建议用 CPU 跑完整数据集：

```powershell
python eval_detectors.py --max-videos 1 --device cpu
```

### 模型下载失败

先确认网络可以访问 Hugging Face；模型成功下载后再进行正式实验。正式论文实验中应记录模型 ID、Transformers 版本、PyTorch 版本、CUDA 版本和 GPU 型号。

### LLM 不可用

单视频 CLI 在没有 `--llm_api_key` 时会使用原始 query 作为检测 query。评估脚本不调用 LLM，直接使用数据集中的英文指代表达。

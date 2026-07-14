**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.
**09/30/2024 -- SAM 2.1 Developer Suite (new checkpoints, training code, web demo) is released**
- A new suite of improved model checkpoints (denoted as **SAM 2.1**) are released. See [Model Description](#model-description) for details.
  * To use the new SAM 2.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, please first uninstall the previous version via `pip uninstall SAM-2`, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.
### Download Checkpoints
First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:
cd checkpoints && \
(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):
Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)) for details on how to add click or box prompts, make refinements, and track multiple objects in videos.
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):
### SAM 2.1 checkpoints
The table below shows the improved SAM 2.1 checkpoints released on September 29, 2024.
### SAM 2 checkpoints
The previous SAM 2 checkpoints released on July 29, 2024 can be found as follows:
The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under [Apache 2.0](./LICENSE), however the [Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file) and [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) used in the SAM 2 demo code are made available under the [SIL Open Font License, version 1.1](https://openfontlicense.org/open-font-license-official-text/).

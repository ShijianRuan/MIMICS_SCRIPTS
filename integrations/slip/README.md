# SLIP — Segmentation with Low-latency Interactive Prompting for 3D Medical Images

[![arXiv](https://img.shields.io/badge/arXiv-2607.22332-b31b1b.svg)](https://arxiv.org/abs/2607.22332)

<!-- <p align="center">
  <img src="assets/model_article_SLIP.drawio.png" alt="SLIP Overview" width="800"/>
</p> -->

**SLIP** is an end-to-end trainable framework for interactive 3D medical image segmentation that decouples image encoding from prompt-guided refinement. Image features are computed once and reused, while a lightweight patch memory bank maintains an interaction-aware segmentation state shared across patches. This persistent representation enables prediction updates by propagating interaction context throughout the image, supports reversible prompting without recomputing image features, and substantially reduces interaction latency.

---

## Installation
Clone the repository and install in editable mode:

```bash
conda env create -n SLIP_env python=3.12
conda activate SLIP_env
git clone https://github.com/your-username/SLIP.git
cd SLIP
pip install -e .
```
### Download Checkpoints

Pretrained model checkpoints are available here:

[Download SLIP Checkpoints](https://cloud.ircad.fr/s/8KW4wa8JGrwm7GP)

### Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| SimpleITK | any |
| TorchIO | any |
| scipy | any |

---

## Quick Start

See [`example_usage.py`](example_usage.py) for a fully commented walkthrough.

```python
from pathlib import Path
from SLIP_model.inference_module import Inference_module

predict = Inference_module(
    device="cuda",
    checkpoint="sam_model_400_clean.pth",
    activate_propagation=True,
    torch_compile=True,
)

# Load and preprocess the volume (computes all patch embeddings)
predict.process_image(
    image_path=Path("my_image.nii.gz"),
    gt_path=Path("my_label.nii.gz"),   # set to None for pure inference
    target_label=1,
)

# First click — coordinates in original image space [z, y, x]
mask = predict.click_inference([[105, 119, 79]], [1])

# Second click
mask = predict.click_inference([[106, 119, 75]], [0])   # 0 = exclude

# Undo the last click
predict.undo(predict.action_history)
```


---

## Reproducing Paper Results

[`results_computing.py`](results_computing.py) implements the **iterative-clicking evaluation protocol** used in the paper.  
It processes a folder of volumes, automatically simulates clicks based on the largest segmentation error region (false negatives / false positives), and records Dice scores and timing for every click.

```bash
python results_computing.py \
    --checkpoint /path/to/checkpoint.pth \
    --images_dir  /path/to/images \
    --labels_dir  /path/to/labels \
    --work_dir     /path/to/output \
    --click_number 10 \
    --labels 1 2 3   # optional: restrict to specific label indices
```

Outputs saved to `work_dir`:
- `results_summary.txt` — per-image Dice progression and timing statistics.
- `results_data.pkl` — full results dictionary for custom analysis.
- `mean_dice_progression.png/.pdf` — publication-ready Dice vs. click-number plot.
- `label_<id>/` — per-label breakdown plots.

---


## License

This project is licensed under the GNU GPL v3.

The modified SAM 2 components remain licensed under the Apache License 2.0.
See LICENSE_sam2

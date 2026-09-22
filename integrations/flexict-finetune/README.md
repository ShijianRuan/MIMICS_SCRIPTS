# FlexiCT Few-Shot Finetuning

A standalone, target-agnostic training framework for **few-shot medical-image
segmentation** with [FlexiCT](https://github.com/ricklisz/FlexiCT) ViT
backbones, on top of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). Supports
both **2D and 3D** segmentation tasks.

This repo distills a single validated recipe — FlexiCT 2D/3D backbone + Primus
multi-scale decoder, few-shot full-finetune in nnU-Net — out of a larger
research codebase. It is independent: no organ-specific names, no leftover
ablation/contrast experiments, no diagnostic scripts. Point it at any CT
target and train.

It also carries a self-contained **uncertainty ranking** module
(`uncertainty/`) for active-learning loops: run two models (e.g. 2D+3D) on
unlabeled data, rank cases by prediction disagreement.

## What's inside

```
flexict-finetune/
├── flexict/                 # FlexiCT backbone + Primus decoder (self-contained pkg)
├── trainers/                # nnU-Net trainers (flexict2d_Trainer / flexict3d_Trainer)
├── scripts/                 # build_dataset / train / predict / evaluate / uncertainty
├── weights/                 # pretrained backbones (flexict_2d, flexict_3d) — not in git
├── configs/                 # example configs
├── uncertainty/             # 2D/3D disagreement uncertainty ranking (numpy/nibabel only)
└── docs/validation.md       # validated results, per-case Dice/HD95
```

## The recipe (validated)

| piece | value |
|---|---|
| backbone | `flexi_ct_backbone_base` — patch 8, embed_dim 864, depth 16, heads 12; 2D & 3D via `PatchEmbedND` dual-mode |
| decoder | `FlexiCTPrimus2D` (ConvTranspose2d ladder) / `FlexiCTPrimus3D` (trilinear+Conv3d ladder); multi-scale token concat over 4 levels `[3,7,11,15]` |
| precision | **fp32** (FlexiCT RoPE overflows in fp16) |
| optimizer | AdamW, 2 groups: backbone `vit_lr=3e-5`, decoder `initial_lr=3e-4`; poly `(1-progress)` |
| data | single-channel CT, CTNormalization, nnU-Net default augmentation (elastic off) |
| nnU-Net | standard `nnUNetPlans`, configs `2d` / `3d_fullres`; `deep_supervision=False`, `oversample_foreground=0.33`, 150 epochs |

Results that validated this recipe (few-shot whole-organ CT, 28-case external
holdout, mean Dice): **2D 0.939 / 3D 0.961**. Per-case tables in
`docs/validation.md`.

## Quick start

### 1. Environment

Pick an nnU-Net env. Two are supported (trainer auto-detects the API version):
- nnunetv2 **2.5.2** + torch 2.1.0 (the validated env)
- nnunetv2 **2.8.0** + torch 2.6.0 (local import/smoke)

```bash
pip install -r requirements.txt
```

### 2. Make the trainer discoverable

**Option A (recommended, nnunetv2 ≥ 2.6):** set the `nnUNet_extTrainer`
environment variable — no site-packages copy needed:

```bash
export nnUNet_extTrainer="$(pwd)/trainers"    # bash
set nnUNet_extTrainer=%CD%\trainers           # Windows cmd
```

`scripts/setup_env.sh` already does this for you. nnU-Net ≥ 2.6 reads this
variable as an extra trainer-discovery path (see
`nnunetv2/paths.py::nnUNet_extTrainer`).

**Option B (nnunetv2 ≤ 2.5.2):** nnU-Net only discovers trainers from its own
package path, so copy the trainer file there (one-time):

```bash
NNUNET_SITE=$(python -c "import nnunetv2,os;print(os.path.dirname(os.path.dirname(nnunetv2.__file__)))")
cp trainers/flexict_trainer.py "$NNUNET_SITE/training/nnUNetTrainer/flexict_trainer.py"
```

### 3. Pretrained weights

`weights/` holds the two pretrained backbones (`flexict_2d/model.safetensors`,
`flexict_3d/model.safetensors`, ~576 MB each, fp32, keys prefixed
`backbone.`). They are **not in git** (~1.1 GB total). Copy them in from your
weight distribution before training — the trainers locate them via
`FLEXICT_EXT_DIR` (repo root) or explicit `FLEXICT2D_CKPT` / `FLEXICT3D_CKPT`
paths (see `trainers/flexict_trainer.py` docstring).

2D and 3D use **different** weight files (a 2D-pretrained and a 3D-pretrained
FlexiCT), even though the backbone Python class is shared; one class serves
both by branching on input rank (4D→2D patch embed, 5D→3D patch embed).

### 4. Build a dataset

```bash
python scripts/build_dataset.py \
    --source  Z:/.../Totalsegmentator_dataset_v201 \
    --label-name liver --dataset-id 907 --dataset-name LiverFS \
    --n-train 8 --out $nnUNet_raw
```
Produces a standard nnU-Net raw dataset (`imagesTr/labelsTr/imagesTs/labelsTs`
+ `dataset.json`, 0/1 single label, RAS canonical).

### 5. Plan + preprocess + train

```bash
source scripts/setup_env.sh
nnUNetv2_plan_and_preprocess -d 907 --verify_dataset_integrity

# 2D
NUM_EPOCHS=150 bash scripts/run_train.sh 907 2d
# 3D
NUM_EPOCHS=150 bash scripts/run_train.sh 907 3d_fullres
```

**GPU memory guidance.** The validated batch sizes come from nnU-Net plans and
must not be overridden (see pitfalls below). As reference points: 2D trains at
batch 8 in ~14 GB; 3D fullres at batch 2 needed ~32 GB on the validation GPU.
On a 12 GB card (e.g. RTX 3060): the **2D config trains fine**; for 3D expect
OOM at default patch size — you may need to reduce the 3D patch size in the
plans (which changes the recipe) or train 3D on a bigger card. When in doubt,
train 2D: in validation it was within 0.02 mean Dice of 3D.

### 6. Predict + evaluate

```bash
bash scripts/run_predict.sh 907 2d <input_dir> <output_dir>
python scripts/evaluate.py --pred <output_dir> --gt <labelsTs> --out eval.json
```

### 7. Uncertainty ranking (active learning)

```bash
python scripts/run_uncertainty.py \
    --mask-dirs preds_2d preds_3d_fullres \
    --out-dir uncertainty_out --method disagreement \
    --target-labels 1
```
Ranks cases by 2D-vs-3D prediction disagreement (integrated score over a
uint8×10 uncertainty map + consensus mask per case, `ranking.csv` with
per-case scores). Use it to decide which unlabeled cases to annotate next.
Requires only numpy/scipy/nibabel — no torch, no GPU.

## Key design choices (why this works under few-shot)

- **fp32, not fp16.** FlexiCT's RoPE position encoding overflows to NaN under
  fp16 autocast (backward pass). The trainers disable `GradScaler` and run the
  forward without autocast.
- **Mirror axes are configurable, and an L-R axis is excludable.** For a
  single-side / asymmetric target, mirroring the lateral axis turns left into
  right and creates contralateral false positives. Set `MIRROR_DISABLE_AXES`
  to drop that axis; defaults keep nnU-Net's standard mirroring otherwise.
- **Backbone/decoder split optimizer.** A 10× lower LR on the pretrained ViT
  backbone vs the randomly-init decoder stabilizes few-shot finetuning.
- **No elastic deformation.** The default nnU-Net `SpatialTransform` elastic
  path was misused (per-axis vs range API), causing a silent data/seg desync
  that produced "ghost organ" oversegmentation. Elastic is left at `p=0`; the
  rest of nnU-Net's augmentation is kept at default.
- **Never override batch size.** Clamping batch size in the trainer
  desynchronized nnU-Net's oversample-foreground schedule and caused OOM in a
  subtle, recipe-breaking way. Change the plans, not the trainer.

See `docs/validation.md` for validated numbers and per-case results.

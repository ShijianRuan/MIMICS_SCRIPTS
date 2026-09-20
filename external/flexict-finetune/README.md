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

## What's inside

```
flexict-finetune/
├── flexict/                 # FlexiCT backbone + Primus decoder (self-contained pkg)
├── trainers/                # nnU-Net trainers (flexict2d_Trainer / flexict3d_Trainer)
├── uncertainty/             # per-voxel uncertainty + case ranking (Action6 method)
├── scripts/                 # build_dataset / train / predict / evaluate / uncertainty
├── weights/                 # pretrained backbones (flexict_2d, flexict_3d)
├── configs/                 # example configs
└── docs/                    # pitfalls.md, validation log, uncertainty.md
```

## The recipe (validated)

| piece | value |
|---|---|
| backbone | `flexi_ct_backbone_base` — patch 8, embed_dim 864, depth 16, heads 12; 2D & 3D **share one weight file** via `PatchEmbedND` dual-mode |
| decoder | `FlexiCTPrimus2D` (ConvTranspose2d ladder) / `FlexiCTPrimus3D` (trilinear+Conv3d ladder); multi-scale token concat over 4 levels `[3,7,11,15]` |
| precision | **fp32** (FlexiCT RoPE overflows in fp16) |
| optimizer | AdamW, 2 groups: backbone `vit_lr=3e-5`, decoder `initial_lr=3e-4`; poly `(1-progress)` |
| data | single-channel CT, CTNormalization, nnU-Net default augmentation |
| nnU-Net | standard `nnUNetPlans`, configs `2d` / `3d_fullres`; `deep_supervision=False`, `oversample_foreground=0.33`, 150 epochs |

Results that validated this recipe (few-shot whole-organ CT, nnU-Net 7-case
internal holdout Dice): **2D 0.956 / 3D 0.960** (left), **3D 0.969 / 2D 0.942**
(right). See `docs/pitfalls.md` for the full set of problems that had to be
solved to get there.

## Quick start

### 1. Environment

Pick an nnU-Net env. Two are supported (trainer auto-detects the API version):
- nnunetv2 **2.5.2** + torch 2.1.0 (the validated env)
- nnunetv2 **2.8.0** + torch 2.6.0 (local import/smoke)

```bash
pip install -r requirements.txt
```

### 2. Install the trainer into nnU-Net

nnU-Net discovers trainers from its own package path, so copy the trainer file
there (one-time):

```bash
NNUNET_SITE=$(python -c "import nnunetv2,os;print(os.path.dirname(os.path.dirname(nnunetv2.__file__)))")
cp trainers/flexict_trainer.py "$NNUNET_SITE/training/nnUNetTrainer/flexict_trainer.py"
```

### 3. Build a dataset

```bash
python scripts/build_dataset.py \
    --source  Z:/.../Totalsegmentator_dataset_v201 \
    --label-name liver --dataset-id 907 --dataset-name LiverFS \
    --n-train 8 --out $nnUNet_raw
```
Produces a standard nnU-Net raw dataset (`imagesTr/labelsTr/imagesTs/labelsTs`
+ `dataset.json`, 0/1 single label, RAS canonical).

### 4. Plan + preprocess + train

```bash
source scripts/setup_env.sh
nnUNetv2_plan_and_preprocess -d 907 --verify_dataset_integrity

# FlexiCT is memory-heavy (fp32, 175M/355M params). nnU-Net's auto batch_size
# is sized for a small PlainConvUNet and will OOM. Lower it in the plans:
python scripts/set_plan_batch_size.py --preprocessed $nnUNet_preprocessed \
    --dataset 907 --config 2d --batch-size 8
python scripts/set_plan_batch_size.py --preprocessed $nnUNet_preprocessed \
    --dataset 907 --config 3d_fullres --batch-size 2

# 2D
NUM_EPOCHS=150 bash scripts/run_train.sh 907 2d
# 3D
NUM_EPOCHS=150 bash scripts/run_train.sh 907 3d_fullres
```

### 5. Predict + evaluate

```bash
bash scripts/run_predict.sh 907 2d <input_dir> <output_dir>
python scripts/evaluate.py --pred <output_dir> --gt <labelsTs> --out eval.json
```

### 6. Uncertainty ranking (find cases where 2D and 3D disagree most)

Once 2D and 3D models are trained, run both on the same inputs and rank every
case by per-voxel disagreement between the two models. Cases that bubble to the
top are where the models disagree most — the natural next targets for review or
re-annotation. See `docs/uncertainty.md` for the method and 2-mask defaults.

```bash
# One command: prepare inputs -> predict 2D + 3D -> rank by disagreement
bash scripts/run_uncertainty_rank.sh <source> 907 0 0 2   # full run, GPU0=2D GPU2=3D
bash scripts/run_uncertainty_rank.sh <source> 907 5 0 2   # smoke (5 cases)

# Streaming: data lives elsewhere, GPU box has no room for all of it —
# stream one case at a time (scp -> predict -> scp back -> remote cleanup),
# then rank locally. Runs locally, drives the remote GPU.
python scripts/stream_uncertainty_rank.py \
    --source <v201_root> --remote user@host -p 11208 \
    --remote-root '~/flexict-finetune' --dataset-id 907 \
    --local-masks ./unc_masks --out ./unc_out --gpu-2d 0 --gpu-3d 2

# Or step-by-step (predictions already produced):
python scripts/run_uncertainty.py \
    --mask-dirs preds_2d preds_3d_fullres \
    --cases-file infer_input/case_list.txt \
    --out uncertainty_out --method disagreement --target-labels 1
```
Output: `uncertainty_ranking.csv` (cases sorted high→low by integrated
uncertainty) + per-case `*_uncertainty_disagreement.nii.gz` maps.

## Key design choices (why this works under few-shot)

- **fp32, not fp16.** FlexiCT's RoPE position encoding overflows to NaN under
  fp16 autocast. The trainers disable `GradScaler` and run the forward without
  autocast. See `docs/pitfalls.md`.
- **Mirror axes are configurable, and an L-R axis is excludable.** For a
  single-side / asymmetric target, mirroring the lateral axis turns left into
  right and creates contralateral false positives. Set `MIRROR_DISABLE_AXES` to
  drop that axis; defaults keep nnU-Net's standard mirroring otherwise.
- **Backbone/decoder split optimizer.** A 10× lower LR on the pretrained ViT
  backbone vs the randomly-init decoder stabilizes few-shot finetuning.
- **No elastic deformation.** The default nnU-Net `SpatialTransform` elastic
  path was misused (per-axis vs range API), causing a silent data/seg desync
  that produced "ghost organ" oversegmentation. Elastic is left at `p=0`; the
  rest of nnU-Net's augmentation is kept at default.

See `docs/pitfalls.md` for the full diagnosis of each issue above.

## Notes

- `weights/` holds the two pretrained backbones (`flexict_2d/model.safetensors`,
  `flexict_3d/model.safetensors`, ~576 MB each). 2D and 3D use **different**
  weight files (a 2D-pretrained and a 3D-pretrained FlexiCT), even though the
  backbone Python class is shared.
- The 2D/3D backbone is the same `Flexi_CT_Backbone`; it branches on input
  rank (4D→2D patch embed, 5D→3D patch embed) inside `get_intermediate_layers`,
  so one class serves both.

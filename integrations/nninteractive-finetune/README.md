# nnInteractive Task Adaptation

This package fine-tunes an installed nnInteractive model on task-specific 3D
images and binary labels. It is an external Python workflow: it does not import
Mimics and cannot block the Mimics GUI.

## Scope

The first validated implementation focuses on click-guided task adaptation:

- Reconstructs the official nnInteractive network from its `plans.json`,
  `dataset.json`, and checkpoint.
- Preserves the eight-channel contract: one image plus seven interaction
  channels.
- Simulates a true sequence of predictions, previous masks, false-negative
  positive clicks, and false-positive negative clicks.
- Supports CLoPA InstanceNorm adaptation (`clopa_in`), CLoPA convolution plus
  normalization adaptation (`clopa_conv`), and full fine-tuning (`full`).
- Writes a normal nnInteractive model folder that the official inference
  session can load directly.
- Reports progress through an atomic JSON status file, supports a cancellation
  marker, resumes compatible interrupted jobs, and prevents concurrent writers
  from using the same output directory.

Mixed point, box, lasso, and scribble training is intentionally not presented
as validated functionality yet. The official inference code supports those
prompts, but open training implementations do not currently provide enough
evidence for a faithful production implementation.

## Data Contract

Each manifest row points to one image and one label NIfTI. The pair must have the
same shape and physical affine after lossless canonical orientation. The loader:

1. Reorients both arrays to canonical RAS using axis permutations and flips only.
2. Rejects shape or affine mismatches. It never hides a mismatch by resampling a
   label.
3. Converts the configured label values to one foreground class.
4. Applies the same nonzero-bounding-box z-score normalization used by the
   current nnInteractive inference session.
5. Caches normalized arrays as `.npy` files to avoid repeatedly decompressing
   `.nii.gz` volumes.

Labels exported from Mimics must first be mapped back to the original image
grid. That responsibility belongs to the Mimics export bridge; this trainer
will fail closed if image and label geometry disagree.

## Install

Use the same external environment that already runs nnInteractive:

```bash
python -m pip install -e external/nninteractive-finetune
```

No separate model conversion is required.

## Audit The Base Model

```bash
python -m nninteractive_finetune audit \
  --model-dir /path/to/nnInteractive_v1.0
```

The command checks the stem channel count, binary output heads, plans, fold, and
checkpoint structure before training.

## Train

Edit `configs/clopa_3060.yaml`, then run:

```bash
python -m nninteractive_finetune train \
  --config external/nninteractive-finetune/configs/clopa_3060.yaml
```

`clopa_3060.yaml` uses `128x128x128` patches, batch size 1, and gradient
accumulation 2 as a memory-conscious starting point. The research-reference
configuration uses `192x192x192` patches and batch size 2; it is unlikely to fit
on a 12 GB GPU without additional memory optimization.

To cancel, create the configured `cancel_path`. If it is blank, create:

```text
<output-parent>/_nninteractive_finetune_work/<output-name>/cancel.request
```

The status file is in the same work directory unless `status_path` is set.

## Evaluate The Interaction Trajectory

```bash
python -m nninteractive_finetune evaluate \
  --model-dir /path/to/adapted_model \
  --manifest /path/to/dataset_manifest.json \
  --output /path/to/evaluation.json \
  --clicks 5
```

Evaluation loads the model once, then uses the real
`nnInteractiveInferenceSession` for every case. It reports Dice after each
simulated corrective click and trajectory AUC. A final Dice alone is not enough
to establish that an interactive model became easier to use.

New Mimics training requests use one of three goals: **General adaptation**
(default), **Start from an empty Mask**, or **Refine an existing Mask**. The
default runs one to five sequential corrections, weighted toward one to three,
and reports checkpoints at 1, 3, and 5 interactions. Every point is followed
by a prediction before the next error is sampled.

The default `official_single` correction policy selects one false-negative or
false-positive connected component, weighted by its volume, and places one
signed point in that component. Empty predictions therefore receive a
foreground point, pure over-segmentations receive a background point, and
mixed errors receive one sampled correction rather than a forced pair.

General and refinement training can use an optional `initial_mask` file in each
manifest row. The Mimics model center exposes three sources: no Initial Mask, a
different Mask in each saved `.mcs` project, or previously exported Initial
Masks. Synthetic Initial Masks are not generated. `start_empty` always begins
from an empty Mask; `refine_existing` only accepts cases with a usable real
draft; `general` can mix real-draft cases with empty-start cases. Empty or
target-identical drafts are rejected explicitly.

Validation reports empty-start and real-Initial-Mask trajectories separately.
For each mode it records the Dice before any click, the mean AUC at the selected
interaction counts, and AUC gain relative to that mode's own baseline. Model
comparison uses empty against empty and real draft against the same real draft;
no model can be auto-selected without paired validation cases, and every
available start mode must meet the configured improvement threshold.

Real Initial Masks are also stratified without extra user input. Preparation
records their source type/model, full-volume baseline Dice, precision, recall,
and volume ratio, and assigns low (`<0.4`), medium (`0.4-0.7`), or high
(`>=0.7`) quality. Training summaries and real-session evaluation reports keep
quality- and source-specific trajectories so a strong average cannot conceal a
regression on rough drafts.

Input preparation uses one explicit contract on both training and deployment:
NIfTI arrays are reindexed to canonical RAS by axis permutation and flipping
only; image, final target, and Initial Mask must have matching shape and affine;
source spacing is preserved and is not used for an extra whole-volume resample;
and normalization matches the official session's nonzero-bounding-box z-score.
The official nnInteractive session then performs prompt-centred crop/resize
according to the model plan during inference. Training uses fixed voxel patches
for memory control, which changes available context but does not introduce a
second spacing or intensity transform.

Model export also writes the training `point_radius` and
`interaction_decay` into `inference_info.json`. This prevents a newer or older
nnInteractive runtime from silently applying different prompt-channel defaults
during inference.

Historical YAML files that contain only `interaction_steps: N` retain an exact
N-step CLoPA-style paired-click budget for experiment reproducibility. New
configurations use `training_goal`, `correction_policy`,
`min_interaction_steps`, `max_interaction_steps`,
`interaction_step_weights`, and `validation_interaction_steps` explicitly. The
default uses CLoPA paired corrections with a moderate 1-8-step long tail and
validates at 1/3/5/8 steps; it deliberately does not run the paper's expensive
100-step research evaluation during routine training.

The default `nninteractive_nnunet` augmentation profile follows the public
nnU-Net training recipe used by nnInteractive's ResEnc-L backbone: 3D
rotation/scaling, Gaussian noise and blur, multiplicative brightness,
contrast, low-resolution simulation, two gamma transforms, and mirroring use
the official probabilities and ranges. The operators are implemented locally
with NumPy/SciPy/PyTorch to keep the offline Windows environment independent of
the private nnInteractive training code. Spatial scaling direction, even-patch
centering, linear image interpolation, nearest label interpolation, reflected
Gaussian padding, and low-resolution interpolation follow the public current
`batchgeneratorsv2` implementation. Binary targets use the public transform's
linear class-probability decision rather than an unrelated smoothing operation.
Image, target, and Initial Mask share exactly the same sampled spatial
transform; intensity transforms affect only the image. This is
mathematical/protocol alignment, not a claim of bitwise RNG equivalence with
the unpublished nnInteractive trainer.

Mimics exposes an `Anatomy mirroring` choice. Its default `Automatic` mode
keeps all three canonical-RAS mirror axes for ordinary targets, but removes RAS
axis 0 when the task or Target Mask name explicitly contains left/right,
L/R, or 左/右. Users can explicitly preserve laterality or allow every axis;
the resolved policy and axes are stored in the training configuration and log.

## Evidence And Boundaries

The architecture and prompt contract follow the
[official nnInteractive repository](https://github.com/MIC-DKFZ/nnInteractive)
and [nnInteractive paper](https://arxiv.org/abs/2503.08373). The parameter
policies and the historical ten-epoch, five-interaction protocol follow
[CLoPA](https://arxiv.org/abs/2603.06426). The public
[segfm3d_nora_team implementation](https://github.com/tidiane-camaret/segfm3d_nora_team)
demonstrates checkpoint reconstruction and click-aware adaptation, but its
wrapper computes multiple clicks from one unchanged prediction. This package
instead recomputes the prediction after every correction step; a CLoPA paired
step adds the available foreground/background pair and then predicts once.

The original nnInteractive training system is not public. Therefore this is a
research-backed task-adaptation implementation, not a claim to reproduce the
foundation model's original pretraining.

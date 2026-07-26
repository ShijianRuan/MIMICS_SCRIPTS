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

## Evidence And Boundaries

The architecture and prompt contract follow the
[official nnInteractive repository](https://github.com/MIC-DKFZ/nnInteractive)
and [nnInteractive paper](https://arxiv.org/abs/2503.08373). The parameter
policies and default ten-epoch, five-interaction protocol follow
[CLoPA](https://arxiv.org/abs/2603.06426). The public
[segfm3d_nora_team implementation](https://github.com/tidiane-camaret/segfm3d_nora_team)
demonstrates checkpoint reconstruction and click-aware adaptation, but its
wrapper computes multiple clicks from one unchanged prediction. This package
instead recomputes the prediction after every correction.

The original nnInteractive training system is not public. Therefore this is a
research-backed task-adaptation implementation, not a claim to reproduce the
foundation model's original pretraining.

# nnInteractive Fine-Tuning Implementation Status

## Objective

Adapt an installed nnInteractive model to one binary annotation task using
images and masks produced during a Mimics annotation campaign. Training runs in
an external Python process and produces a versioned model folder that the
normal nnInteractive inference session can load without a conversion layer.

This document separates implemented behavior from research hypotheses and
future Mimics integration work.

## Implemented

### Official Model Compatibility

- Reconstructs the network with the current
  `nnInteractiveTrainer_stub.build_network_architecture` API.
- Derives image channels from the model plans and adds seven interaction
  channels.
- Forces the binary two-logit output expected by nnInteractive.
- Audits the checkpoint stem, output heads, plans, fold, and state dictionary.
- Preserves model metadata and license files in the adapted model folder.
- Removes the base `network_weights` dictionary from live trainer metadata after
  loading, avoiding a permanent host-memory duplicate of approximately 392 MB.

### Data And Geometry

- Accepts explicit image and label paths through a dataset manifest.
- Uses valid NIfTI qform/sform geometry and rejects files when both transforms
  are valid but conflict.
- Reorients image and label to canonical RAS with permutations and flips only.
  This operation does not interpolate voxels.
- Requires equal shape and affine after canonical orientation.
- Never silently resamples a label.
- Converts selected source values into one binary foreground class.
- Uses nnInteractive's nonzero-bounding-box statistics to z-score the complete
  image.
- Materializes prepared arrays once and reuses memory maps.
- Caches a bounded foreground-center index so every training patch does not
  rescan the full label volume.

### Interactive Adaptation

- Supports `clopa_in`, `clopa_conv`, and `full` trainability policies.
- Uses fixed 3D patches with configurable foreground/background sampling.
- Creates the same radius-four discrete EDT point kernel as the official
  inference implementation.
- Runs a real sequence of:
  1. current prediction;
  2. previous-segmentation channel update;
  3. false-negative positive click;
  4. false-positive negative click;
  5. next prediction.
- Decays older prompt channels by the model's legacy interaction rule.
- Averages Dice plus cross-entropy over interaction steps.
- Stops including a sample in later interaction losses once it reaches perfect
  Dice, as specified by CLoPA.
- Applies paired spatial flips and conservative intensity perturbations to
  training patches. Validation data is not augmented.

### Lifecycle And Diagnostics

- Writes an atomic status JSON with phase, epoch, update, loss, Dice, elapsed
  time, trainable parameter count, completion, cancellation, or failure.
- Falls back to a flushed direct status write when Windows or SMB permits file
  writes but repeatedly denies atomic replacement.
- Uses a cancellation marker checked during data preparation and every gradient
  update.
- Uses an exclusive output lock and safely reclaims a stale lock only when its
  owner process is no longer alive.
- Saves resumable optimizer, random-number, trainable-parameter, best-parameter,
  and history state.
- Rejects a resume state when the base checkpoint or relevant configuration
  changed.
- Retries checkpoint publication when Windows temporarily denies `os.replace`.
- Loads one official inference session for an entire evaluation run instead of
  rebuilding the model per case.

## Verified Locally

The following checks were completed on 2026-07-24:

| Check | Result |
|---|---|
| Automated unit and workflow tests | 16 passed |
| Synthetic NIfTI train workflow | Completed with two sequential interactions |
| Official checkpoint audit | Compatible |
| Official network reconstruction | Successful |
| Official model parameters | 102,355,818 |
| `clopa_in` trainable parameters | 27,648 |
| `clopa_conv` trainable parameters | 145,216 |
| Official and local point kernels | Bit-identical |
| Adapted-folder export audit | Compatible |
| Load exported folder with installed nnInteractive 2.4.2 | Successful |
| Python wheel build | Successful |

The local machine did not run a real clinical-data GPU convergence experiment.
The synthetic workflow validates control flow, gradients, state, geometry
checks, and lifecycle behavior; it does not establish task accuracy.

## Deliberately Deferred

### Mixed Prompt Training

The official model supports points, boxes, lassos, and scribbles during
inference. The open training examples currently available do not provide a
production-quality, current-API implementation that faithfully reproduces the
foundation model's mixed-prompt curriculum. The first release therefore trains
click interaction only, matching the CLoPA study. Mixed prompts must be added
behind an experimental flag and validated by prompt-specific trajectory
ablations before becoming a user option.

### Automatic Episode Scheduling

CLoPA proposes adaptation episodes as the annotation cache grows. Scheduling is
not embedded in the independent trainer because the trainer does not own the
Mimics annotation queue. The Mimics integration should decide when to create a
new immutable model version and launch this package with a frozen manifest.

### GPU Resource Arbitration

The trainer owns only its output-directory lock. The later Mimics launcher must
acquire the project's shared GPU resource lock before starting training, so an
nnInteractive inference server and training process cannot compete for the
same 12 GB GPU. A cancellation request must terminate and reap the training
process before that shared lock is released.

## Mimics Integration Contract

The integration layer should remain thin:

1. Let the user select a task/mask name and specific annotated cases.
2. Export each selected mask back to its source-image physical grid.
3. Write an immutable dataset manifest containing explicit image and label
   paths.
4. Validate the manifest with the independent trainer before reserving the GPU.
5. Launch training with the external nnInteractive Python environment.
6. Poll the status JSON with a Mimics timer; never wait synchronously.
7. Expose progress, cancel, retry, and the final interaction-trajectory report.
8. Register the completed model as a new version for the task. Never overwrite
   the previous successful model automatically.
9. Pass the selected model directory directly to the existing nnInteractive
   server.

The export bridge is responsible for mapping Mimics masks to the original image
grid. The trainer's strict affine check is the final guard against training on
misregistered labels.

## Research Basis

- [nnInteractive repository](https://github.com/MIC-DKFZ/nnInteractive)
- [nnInteractive paper](https://arxiv.org/abs/2503.08373)
- [CLoPA](https://arxiv.org/abs/2603.06426)
- [segfm3d_nora_team reference implementation](https://github.com/tidiane-camaret/segfm3d_nora_team)
- [nnU-Net pretraining and fine-tuning constraints](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md)

The public segfm wrapper is useful evidence for checkpoint reconstruction and
click-aware training, but it generates multiple clicks from one unchanged
prediction and targets an older nnU-Net API. Those limitations are not carried
into this implementation.

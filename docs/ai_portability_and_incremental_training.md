# AI Data Portability and Incremental Training

## Scope

This document defines how DINOv3 and nnInteractive task-model workflows behave
when data or models move between workstations, and what is reused when a later
training run changes only part of a dataset.

## What Can Move

### Complete dataset folder

Copying the complete dataset folder is the preferred migration method. Keep
these items together:

```text
dataset/
  <case-id>/...
  mcs_output/
    <case-id>.mcs
    dataset_manifest.json
  fewshot_models/
```

`dataset_manifest.json` stores both a relative and an original absolute path.
Resolution prefers the relative path. A moved dataset therefore remains usable
when the same internal directory structure is retained, even though the old
drive letter or workstation path no longer exists.

The DINOv3 prediction and status entries no longer ask for a dataset folder at
startup:

- Prediction resolves the open project, manifest, source image, recent
  workspaces, and reusable model registry automatically.
- Custom nnInteractive inference also uses the manifest when the absolute
  source path stored in `.mcs` metadata no longer exists.
- Status and Stop resolve the most recently active known workspace.
- If no history exists, Status opens a user-local model-management workspace
  so a portable model package can be imported without selecting a dataset.
- A path error is shown only when the current project cannot be linked safely.

### Model packages

Do not rely on copying a machine-local registry JSON. Use the model package
controls in the DINOv3 status window or nnInteractive model center.

The package contains relative model artifacts and portable inference
configuration. Import creates a new local registry entry whose paths point to
the destination workstation. Original training images are not required for
inference once the target case's source image is available.

### Only the `.mcs` file

A copied `.mcs` can still be opened and edited in Mimics. It is not, by itself,
a verified replacement for the source image used to train a source-grid AI
model.

DINOv3 and custom nnInteractive models require the declared physical-intensity
and geometry input contract. Mimics voxel buffers can differ in intensity
encoding and grid from the source image. The integration therefore does not
silently use a display/GV buffer when a source-trained model expects the source
grid.

If the source path is missing after migration:

1. Restore the source image under the copied dataset structure, or select that
   dataset/source root in the training data interface.
2. Keep or rebuild `dataset_manifest.json` beside the `.mcs` files.
3. Confirm that source shape and affine match the geometry stored in the
   project metadata before prediction.

Status viewing, stopping a known task, model import/export, and ordinary Mimics
editing do not require the original source image.

## Incremental Reuse

### Local `.mcs` label export

Both DINOv3 and nnInteractive keep a persistent label cache under their
workspace:

```text
<workspace>/cache/mcs_labels/<task-or-organ>/<case-id>/
```

The cache key includes the source image identity, `.mcs` identity, requested
Mask names, and export contract version. A later training run:

- reuses labels for unchanged cases;
- exports only new or changed `.mcs` cases;
- invalidates a case when its image, project, Mask selection, or export
  contract changes;
- keeps per-run staging temporary and deletes it after publication.

DINOv3 additionally keeps source-grid image/label pairs under:

```text
<workspace>/cache/materialized/<organ>/<case-id>/
```

It hard-links or copies unchanged cached pairs into the current run. A changed
source image or label rebuilds only that case. If the optional cache cannot be
written, training falls back to direct materialization and records a warning
instead of failing solely because the optimization was unavailable.

nnInteractive also reuses prepared image/label arrays under:

```text
<workspace>/cache/prepared_cases/<task-id>/
```

Prepared-cache keys use the stable source image and source label or `.mcs`
identity rather than a temporary job path.

### Remote upload

Remote training materializes the same source-grid data locally, then creates
one deterministic archive per case. Each case is addressed by its SHA-256:

```text
<remote-root>/cache/<ssh-user>/datasets/<case-sha256>.tar
```

Only new or byte-changed cases are uploaded. The visible progress is aggregated
over all selected cases and does not reset between files. The small job request
is always uploaded because it contains the current split and training options.

## nnInteractive Interaction Distribution

Training no longer assumes exactly five interactions. Each sample receives an
independent interaction budget:

| Profile | Training range | Validation checkpoints |
|---|---:|---|
| Quick correction | 1-3 | 1, 3 |
| Typical annotation | 1-10 | 1, 3, 5, 10 |
| Extended refinement | 1-15 | 1, 3, 5, 10, 15 |

The default is **Typical annotation**. Short sequences are sampled more often,
while longer sequences remain represented. Training stops early for a sample
when no correction remains.

An interaction is driven by the current error:

- false-negative voxels permit a foreground point;
- false-positive voxels permit a background point;
- a point type is not invented when its corresponding error is absent.

Some samples start from a deliberately imperfect synthetic Mask in the
previous-segmentation channel. Validation reports separate trajectories for an
empty start and an existing-Mask start. This trains and measures the real
workflow where a user continues correcting an existing Mask instead of always
starting from an empty result.

## Remote Runtime Boundary

The saved server profile contains the Linux host/IP, SSH port, SSH username,
authentication method, persistent remote work folder, runtime image, and GPU
selection.

The client connects to the host SSH service. The host starts a disposable
Docker container for each job. The container:

- does not run SSH or expose an SSH port;
- uses the selected GPU only;
- mounts administrator-provided base weights read-only;
- extracts selected case archives into the job directory;
- writes status, metrics, and logs to persistent host storage;
- is removed after confirmed success or cancellation.

Failed job files remain available for diagnosis. The Docker image and base
weights remain installed, and verified case caches remain available for the
next run.

# Mimics Scripting Library Workflows

## Design goals

- A user action must return control to the Mimics GUI quickly. Long-running discovery, conversion, AI inference, training, and status viewing run outside the Mimics process.
- Successful automatic actions remain visible. A completed automatic mask application, training or prediction completion, and a successful stop request use non-blocking notifications.
- Startup acknowledgements, ordinary cancellation, and duplicate confirmations are logged instead of shown as Mimics dialogs.
- Errors that require user action and destructive confirmations remain visible.

## Final entry layout

### 01 Data

1. `01 Import Data`
2. `02 Import Masks`
3. `03 Export Masks`
4. `04 Task Status`

`01 Import Data` opens the always-on-top import window. Drop a single
`.nii`, `.nii.gz`, `.mha`, `.mhd`, or `.nrrd` volume, a TotalSegmentator-style
case folder, a `dicom/` case folder, a DICOM series folder, a multi-selection
of case folders, or a whole dataset folder onto it — or paste paths (Ctrl+V,
one per line) or pick them with the choose buttons. The window classifies the
payload, resolves it against the dataset profile, and submits it to the
background import workers; header inspection and conversion run in the
external bridge process. The window closes itself after an idle timeout.

Mimics starts that process and returns immediately; file browsing never runs
in the Mimics GUI process. Import always requires a source and shows the
resolved `.mcs` output folder, while keeping the output optional to change.
`mimics_io_config.json:mimics_output_dir` is only an administrator/default
value and is overridden by the current window selection. Paths are remembered
only when the annotator explicitly enables the workstation remember option.

`02 Import Masks` adds binary or multi-label NIfTI, MHA/MHD, or NRRD
segmentations to the active image. File reading, label splitting, and spatial
resampling run in external Python. Mimics applies one prepared Mask per GUI
timer tick; empty results are rejected with a spatial-alignment warning.

`03 Export Masks` exports every Mask in the saved project, including hidden
Masks. Select the source image or case and then the destination root. The source
may be NIfTI, MHA/MHD, NRRD, one DICOM file, a DICOM series folder, or a case
folder containing one of those forms. Output is written to `<chosen
root>/<case>/segmentations/*.nii.gz`. Choose `Skip existing` or `Overwrite
existing` in the same external setup window. The currently open `.mcs` path is
read from Mimics and passed to the background process automatically; it is not
a user-configurable data path.
`mimics_output_dir` never redirects exported labels. External batch export uses
`tools/mimics_batch_cli.py export-labels` and requires an explicit destination
policy through either `--output-dir` or `--overwrite-source`.

`04 Task Status` aggregates every import run, background `.mcs` queue,
mask-export job, foreground export task, mask-append job, and drop import into
one live table. Each running row has a Stop button that writes the same
kind-specific stop marker the task's own worker polls: the current case
finishes and no new case starts. It does not stop AI training or inference,
nnInteractive, foreground Mimics, or unrelated processes.

### 02 AI

1. `01 nnInteractive`
2. `Annotate With ScribblePrompt`
3. `FlexiCT/01 Train Model`
4. `FlexiCT/02 Predict Current Case`
5. `FlexiCT/03 Active Learning Review`
6. `FlexiCT/04 Show Status and Stop`
7. `nnUNet/01 Train Model`
8. `nnUNet/02 Predict Current Case`
9. `nnUNet/03 Show Status & Models`

The training and prediction entries remain separate so the common path keeps a
dedicated setup window while prediction stays one click.

### 03 Review

1. `01 Identify Mask At Cursor`
2. `02 Window From Selected Mask`
3. `03 Window Choose Preset`
4. `04 Window Undo Last`
5. `05 Window Edit Presets`

### 99 Admin

1. `01 Setup Repair Environment`
2. `02 Clear Cache`
3. `03 Stop All Owned Services`
4. `04 Undo Last Import`
5. `05 System Health`
6. `06 Collect Diagnostics`
7. `07 Edit Configs`
8. `08 Manage AI Models`
9. `09 Environment Guidance`

The administrative stop entry only targets processes created and registered by this project. The import-specific stop remains under Data because it is part of the import workflow.

## nnInteractive result policy

When a selected Mask already contains segmentation, nnInteractive uses it as an immutable source snapshot. The first prediction runs without an output-target prompt. When that result is ready, one dialog lets the annotator update the selected Mask or create a new `<source> - AI Draft`; the same dialog is the completion notification.

An empty selected Mask and an existing AI Draft are refined in place. If no Mask is selected, an empty `nnInteractive Result` Draft is created automatically. An unused automatically created Draft is removed when prompt collection is cancelled.

The source Mask is exported once for the initial AI session. A Draft is created only after the user chooses `Create Editable Copy`, avoiding a synchronous full-volume Mask copy in Mimics. Session metadata records the source identity, source hash, target identity, and expected target hash to prevent applying a stale result to the wrong Mask.

Set `existing_mask_result_mode` to `in_place` or `derived_copy` only when a fixed non-interactive policy is required. The default is `ask`, with the choice deferred until the first result is ready.

## Notification policy

| Event | User feedback |
| --- | --- |
| nnInteractive prediction ready | One target-choice dialog that also reports completion; no follow-up success dialog |
| Training completed | Non-blocking completion dialog and persistent job log |
| Background task accepted or external window opened | Mimics log only |
| User cancels a picker or setup window | Mimics log only |
| Stop request accepted | Mimics log while cleanup runs; one completion dialog after resources are released |
| Actionable error or destructive decision | Visible dialog with a concrete next step |

## Resource coordination

| Feature | Resource | Coordination policy | User-visible behavior |
| --- | --- | --- | --- |
| Dataset preparation | CPU, disk | Runs independently in external Python | Mimics remains usable |
| `.mcs` creation | One background Mimics instance per active queue | Only creators consuming the same import queue are serialized | Import queue continues automatically |
| Training label export | Saved `.mcs` source and fresh-label destination | Waits for a writer of the same `.mcs` folder or another exporter to the same label destination | Status names the conflicting resource and its owner |
| Current-project mask read/write | Foreground Mimics voxel-buffer API | Mask import, export, and AI result apply share a short `mask_buffer_access` lease | A competing callback is deferred or refused rather than mutating the same masks concurrently |
| nnInteractive prediction | GPU | Serialized with nnU-Net training and prediction | An active prompt is never interrupted |
| Idle nnInteractive worker | GPU | Receives a graceful close request when another AI task needs the GPU | The waiting task starts after release; the next prompt creates a fresh worker |
| nnU-Net training and prediction | GPU | Serialized to prevent CUDA out-of-memory failures | Waiting state, owner, PID, cancellation action, and progress remain visible |

Import does not block AI computation by design. Background Mimics ownership
is scoped to an import queue or export destination, so independent jobs may run
in parallel. Set `MIMICS_SERIALIZE_BACKGROUND_MIMICS=1` only when a workstation's
license or Mimics installation genuinely allows one background instance.

The Mimics `Import Data` entry and
`tools/mimics_batch_cli.py prepare-import` publish the same descriptor format to
the same output-scoped `prepared_queue`. An `import_producer_<scope>.lock`
prevents two of those entry points from changing that queue's active/done state
at the same time. It is released as soon as preparation has committed its work;
the background Mimics consumer may keep creating `.mcs` files afterward. A
different output folder has a different producer and consumer lock and may run
in parallel.

Mask export uses the inverse contract. Every entry first records or measures the
live Mimics voxel-to-RAS matrix, restores the Mimics buffer order, and maps the
label to the selected source-image grid. Internal batch export, external
`export-labels`, and training label staging hold both the directory containing the
source `.mcs` projects and the directory they actually write. This prevents an
exporter from opening a project while an importer is replacing it, and prevents
two exporters from writing the same labels. Locks are acquired in stable path
order. Jobs with unrelated source and destination directories may still run in
parallel.

### Geometry contract shared by every import entry

`Import Data`, `tools/mimics_batch_cli.py
prepare-import`, and `tools/single_case_import_worker.py` all call the same
`mimics_bridge.py` preparation actions and publish the same prepared descriptor.
They do not have separate image-orientation implementations.

The single-case path delegates preparation and `.mcs` waiting to
`single_case_import_worker.py`. The batch path keeps only timer
orchestration in Mimics and runs discovery/conversion in the external Python.
The CLI keeps all orchestration outside Mimics. Despite those lifecycle
differences, all three use the same local runtime root, output-scoped queue,
producer lock, bridge parameters, buffer mapping defaults, stop marker, and
background creator. Starting one entry cannot bypass a task started by another.

Batch preparation is streamed by both the interactive entry and the external
CLI. As soon as one case is prepared, its descriptor is committed and a
background Mimics process can create that `.mcs` while later cases are still
being prepared. The first project therefore waits only for discovery, first-case
preparation, and its own Mimics import/save; it does not wait for the entire
dataset to finish preparation.

A NumPy/NIfTI array has index order, not an anatomical coordinate system. The
NIfTI affine supplies that meaning. For a mask voxel index `s`, its physical
point is `p_RAS = A_mask_RAS * s`. DICOM and Mimics use DICOM patient
coordinates (LPS), so the bridge converts a measured Mimics point to RAS with
`diag(-1, -1, 1, 1)`. Mapping a Mimics target voxel `t` to the source mask is
therefore:

```text
s = inverse(A_mask_RAS) * A_mimics_RAS * t
```

This is one physical-coordinate mapping, not "flip the mask to LPS and then
flip it back." A pure axis permutation, sign flip, or integer translation is
applied exactly. Nearest-neighbor sampling is used only when the physical grids
genuinely differ, so labels remain discrete.

Supported source volumes are `.nii`, `.nii.gz`, `.mha`, `.mhd`, `.nrrd`, and
`.nrrd.gz`, plus a folder containing a DICOM image series. A single `.dcm` file
is not treated as a complete volume; select the series folder instead.

NIfTI headers are checked before SimpleITK reads image data. Matching valid
qform/sform headers use the fast direct-read path. If either form is uncoded,
invalid, or conflicts with the other, the bridge selects a valid coded sform
first, then a valid coded qform, synchronizes a temporary header copy, and reads
that copy. The source file is never rewritten. The same selected affine is used
when NIfTI masks are mapped, preventing image and label readers from choosing
different forms.

For an original DICOM folder, the series is imported directly and its LPS
geometry is converted to RAS only for the bridge's matrix arithmetic. For a
NIfTI, MHD, MHA, or NRRD image, a lossless orientation step may permute or flip
the image indexes before derived DICOM is written. `DICOMOrient("LPS")` changes
the index representation, not the physical location of any voxel: axis-aligned
coronal or sagittal data can become LPS index order by transpose/flip, while a
residual oblique direction remains in the affine. The original source shape and
affine remain recorded separately. The current `auto` policy preserves axial,
coronal, sagittal, orthogonal oblique, and regular gantry-tilt grids. Gantry
tilt is represented by each slice's full `ImagePositionPatient`; it is not an
image resample. Only a non-orthogonal in-plane row/column basis, which classic
DICOM cannot encode, is resampled once to a nearby orthonormal oblique grid.
The older `auto` policy also required the slice step to be perpendicular to the
image plane and therefore unnecessarily resampled regular gantry tilt.

After DICOM import, the background creator measures the live Mimics grid with
`ImageData.get_voxel_center()`. If Mimics has normalized the grid, masks are
mapped directly from their original files to that measured grid; an intermediate
mask is not resampled again. Export performs the inverse mapping from the
measured Mimics grid to the selected original image shape and affine. Thus a
round trip targets the original image grid even when the temporary Mimics grid
differs.

### Switching between features

| Current work | May start immediately | Must wait or stop first | Normal stop | Emergency stop |
| --- | --- | --- | --- | --- |
| Dataset scan or image preparation | Review, window controls, nnInteractive, AI inference on prepared data | Another import of the same queue | `01 Data/04 Task Status` (row Stop) | `99 Admin/03 Stop All Owned Services` |
| Background `.mcs` creation | Review, GPU AI work, and exports reading other `.mcs` folders | Another creator or exporter using the same `.mcs` folder | `01 Data/04 Task Status` (row Stop) | `99 Admin/03 Stop All Owned Services` |
| Mask export | Review, GPU AI work, and imports/exports using unrelated source and destination folders | Import writing its `.mcs` source folder, or export writing the same label destination | `01 Data/04 Task Status` (row Stop) | `99 Admin/03 Stop All Owned Services` |
| nnInteractive active prediction | Review and data preparation | nnU-Net GPU execution | Finish or cancel the current nnInteractive session | `99 Admin/03 Stop All Owned Services` |
| nnInteractive idle image worker | All review and data work | Nothing; another AI task requests a graceful GPU release | Worker exits on idle timeout | `99 Admin/03 Stop All Owned Services` |
| nnU-Net training or prediction | Review, import preparation, status viewer | Another GPU AI task; nnInteractive GPU execution | `02 AI/nnUNet/03 Show Status & Models` (stop from the status window) | `99 Admin/03 Stop All Owned Services` |
| Environment repair | Review and data work | A second environment repair | Re-run the entry and choose `Stop Current Setup` | `99 Admin/03 Stop All Owned Services` |

Resource conflicts fail or wait explicitly; they do not start a competing
writer silently. A background Mimics job may own two scoped
`background_mimics_<scope>.lock` files, one for its source and one for its
destination. `gpu.lock` and these scoped lock files
contain the owner and PID shown in the Mimics log/status UI. Dead-PID locks are
removed automatically. The legacy global `background_mimics.lock` is used only
when explicit single-instance serialization is enabled.
The global stop entry first detaches in-process Mimics timers and asks queues to
stop, then terminates only processes whose command line contains both this
project root and a dedicated Mimics-Script marker. It never targets the
foreground Mimics PID or unrelated Python/Mimics processes.

The default nnInteractive image worker idle timeout is 15 minutes. Its owned inference server exits five minutes after the worker closes. GPU contention can close an idle worker earlier after a short grace period, but cannot close a worker whose state reports an active prediction.

## Storage and retention

Generated files are separated into disposable runtime data and durable user data.

### Automatically bounded or removed

- nnInteractive terminal async jobs: three days, at most 20 recent terminal jobs.
- nnInteractive source-image caches: seven days, at most 12 fingerprinted entries.
- nnInteractive, import, export, and pipeline logs: size-based rotation for shared log files.
- Fresh `.mcs` label staging: removed when the training process exits, including failures and cancellations.
- Failed or cancelled run folders: seven days.
- Completed run text logs: 30 days. Structured metrics and configuration remain available.
- Prediction NIfTI and bridge buffers: removed immediately after successful application to Mimics. Failed results are retained temporarily for diagnosis.

### Never deleted automatically

- Saved `.mcs` projects and exported labels requested by the user.
- Registered model weights, model manifests, configuration, and structured convergence metrics.
- Active job state, active resource locks, and active worker directories.
- Source datasets and original medical images.

`99 Admin > 02 Clear Cache` runs outside the GUI thread. It no longer treats the entire `.mimics_runtime` directory as a cache; active lifecycle state and locks are preserved. `03 Stop All Owned Services` requires both a project-owned path and a dedicated process marker, and explicitly excludes the foreground Mimics process.

The retention defaults are configured in `nninteractive_config.json`. Durable models require deliberate model-version management and are not silently removed based on age.

## Validation still requiring Mimics

The automated fake-Mimics tests verify routing, Draft creation, source preservation, stale-result guards, automatic result notification, and cleanup. A real Mimics installation is still required to verify GUI responsiveness, Mask creation and deletion API behavior, selection state, and the final Scripting Library ordering.

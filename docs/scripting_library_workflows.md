# Mimics Scripting Library Workflows

## Design goals

- A user action must return control to the Mimics GUI quickly. Long-running discovery, conversion, AI inference, training, and status viewing run outside the Mimics process.
- Successful automatic actions remain visible. A completed automatic mask application, training or prediction completion, and a successful stop request use non-blocking notifications.
- Startup acknowledgements, ordinary cancellation, and duplicate confirmations are logged instead of shown as Mimics dialogs.
- Errors that require user action and destructive confirmations remain visible.

## Final entry layout

### 01 Data

1. `01 Import Dataset`
2. `02 Import Single Case`
3. `03 Export Masks`
4. `04 Stop Import Queue`
5. `05 Import Masks`

`02 Import Single Case` accepts a single `.nii`, `.nii.gz`, `.mha`, `.mhd`, or
`.nrrd` volume, a TotalSegmentator-style case folder, a `dicom/` case folder,
or a folder containing a DICOM series directly. Header inspection and
conversion run in the external bridge process.

`01 Import Dataset`, `02 Import Single Case`, and `03 Export Masks` open one
shared external PySide6 path window. Mimics starts that process and returns
immediately; file browsing never runs in the Mimics GUI process. Import always
requires a source and shows the resolved `.mcs` output folder, while keeping the
output optional to change. `mimics_io_config.json:mimics_output_dir` is only an
administrator/default value and is overridden by the current window selection.
Paths are remembered only when the annotator explicitly enables the workstation
remember option.

`05 Import Masks` adds binary or multi-label NIfTI, MHA/MHD, or NRRD
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

### 02 AI

1. `01 nnInteractive`
2. `DINOv3/01 Train Model`
3. `DINOv3/02 Predict Current Case`
4. `DINOv3/03 Predict Choose Model`
5. `DINOv3/04 Show Status Results`
6. `DINOv3/05 Stop AI Task`

The two prediction entries intentionally remain separate. The common path uses the latest valid model directly, while the second path lets the user select a previous model version. Combining them would either remove model-version control or add an unnecessary selection step to every prediction.

### 03 Review

1. `01 Identify Mask At Cursor`
2. `02 Window From Selected Mask`
3. `03 Window Choose Preset`
4. `04 Window Undo Last`
5. `05 Window Reset Full Range`

### 99 Admin

1. `01 Setup Repair Environment`
2. `02 Clear Cache`
3. `03 Stop All Owned Services`

The administrative stop entry only targets processes created and registered by this project. The import-specific stop remains under Data because it is part of the import workflow.

## nnInteractive result policy

When a selected Mask already contains segmentation, nnInteractive uses it as an immutable source snapshot. The first prediction runs without an output-target prompt. When that result is ready, one dialog lets the annotator update the selected Mask or create a new `<source> - AI Draft`; the same dialog is the completion notification.

An empty selected Mask and an existing AI Draft are refined in place. If no Mask is selected, an empty `nnInteractive Result` Draft is created automatically. An unused automatically created Draft is removed when prompt collection is cancelled.

The source Mask is exported once for the initial AI session. A Draft is created only after the user chooses `Create Editable Copy`, avoiding a synchronous full-volume Mask copy in Mimics. Session metadata records the source identity, source hash, target identity, and expected target hash to prevent applying a stale result to the wrong Mask.

Set `existing_mask_result_mode` to `in_place` or `derived_copy` only when a fixed non-interactive policy is required. The default is `ask`, with the choice deferred until the first result is ready.

## Notification policy

| Event | User feedback |
| --- | --- |
| nnInteractive or few-shot prediction ready | One target-choice dialog that also reports completion; no follow-up success dialog |
| Training completed | Non-blocking completion dialog and persistent job log |
| Background task accepted or external window opened | Mimics log only |
| User cancels a picker or setup window | Mimics log only |
| Stop request accepted | Mimics log while cleanup runs; one completion dialog after resources are released |
| Actionable error or destructive decision | Visible dialog with a concrete next step |

## Resource coordination

| Feature | Resource | Coordination policy | User-visible behavior |
| --- | --- | --- | --- |
| Dataset preparation | CPU, disk | Runs independently in external Python | Mimics remains usable |
| `.mcs` creation | One background Mimics process and license | Serialized with label export | Import queue continues automatically |
| DINOv3 label export | One background Mimics process and license | Waits for active `.mcs` creation | Status names the lock owner and points to `04 Stop Import Queue` |
| nnInteractive prediction | GPU | Serialized with DINOv3 training and prediction | An active prompt is never interrupted |
| Idle nnInteractive worker | GPU | Receives a graceful close request when DINOv3 needs the GPU | DINOv3 starts after release; the next prompt creates a fresh worker |
| DINOv3 training and prediction | GPU | Serialized to prevent CUDA out-of-memory failures | Waiting state, owner, PID, cancellation action, and progress remain visible |

Import does not block DINOv3 computation by design. Only the label-export stage waits when import is using the background Mimics license. Users can wait, stop the import queue, or disable fresh label export when already exported labels are intentionally being used.

### Switching between features

| Current work | May start immediately | Must wait or stop first | Normal stop | Emergency stop |
| --- | --- | --- | --- | --- |
| Dataset scan or image preparation | Review, window controls, nnInteractive, DINOv3 inference on prepared data | Another import of the same queue | `01 Data/04 Stop Import Queue` | `99 Admin/03 Stop All Owned Services` |
| Background `.mcs` creation | Review and GPU AI work | Mask export or fresh DINOv3 label export | `01 Data/04 Stop Import Queue` | `99 Admin/03 Stop All Owned Services` |
| Mask export | Review and GPU AI work | Import `.mcs` creation or another label export | `99 Admin/03 Stop All Owned Services` | Same entry; it targets project-owned processes only |
| nnInteractive active prediction | Review and data preparation | DINOv3 GPU execution | Finish or cancel the current nnInteractive session | `99 Admin/03 Stop All Owned Services` |
| nnInteractive idle image worker | All review and data work | Nothing; DINOv3 requests a graceful GPU release | Worker exits on idle timeout | `99 Admin/03 Stop All Owned Services` |
| DINOv3 training or prediction | Review, import preparation, status viewer | Another DINOv3 task for the same dataset; nnInteractive GPU execution | `02 AI/DINOv3/05 Stop AI Task` | `99 Admin/03 Stop All Owned Services` |
| Environment repair | Review and data work | A second environment repair | Re-run the entry and choose `Stop Current Setup` | `99 Admin/03 Stop All Owned Services` |

Resource conflicts fail or wait explicitly; they do not start a competing
process silently. `gpu.lock` and `background_mimics.lock` contain the owner and
PID shown in the Mimics log/status UI. Dead-PID locks are removed automatically.
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
- DINOv3 materialized `imagesTr`, `labelsTr`, `imagesVal`, and `labelsVal`: removed when the training process exits unless explicitly retained.
- Fresh `.mcs` label staging: removed when the training process exits, including failures and cancellations.
- DINOv3 training experiment checkpoints: the deployable checkpoint is copied into the model registry; duplicate experiment artifacts are removed when the process exits by default.
- DINOv3 terminal job records: 30 days, at most 100 recent records.
- Failed or cancelled run folders: seven days.
- Completed run text logs: 30 days. Structured metrics and configuration remain available.
- DINOv3 prediction NIfTI and bridge buffers: removed immediately after successful application to Mimics. Failed results are retained temporarily for diagnosis.

### Never deleted automatically

- Saved `.mcs` projects and exported labels requested by the user.
- Registered DINOv3 model weights, model manifests, configuration, and structured convergence metrics.
- Active job state, active resource locks, and active worker directories.
- Source datasets and original medical images.

`99 Admin > 02 Clear Cache` runs outside the GUI thread. It no longer treats the entire `.mimics_runtime` directory as a cache; active lifecycle state and locks are preserved. `03 Stop All Owned Services` requires both a project-owned path and a dedicated process marker, and explicitly excludes the foreground Mimics process.

The retention defaults are configured in `nninteractive_config.json` and `fewshot_config.json`. Durable models require deliberate model-version management and are not silently removed based on age.

## Validation still requiring Mimics

The automated fake-Mimics tests verify routing, Draft creation, source preservation, stale-result guards, automatic result notification, and cleanup. A real Mimics installation is still required to verify GUI responsiveness, Mask creation and deletion API behavior, selection state, and the final Scripting Library ordering.

# DINOv3 Few-Shot Mimics Integration Design

## Goals

This integration adds a few-shot organ segmentation workflow to Mimics without running PyTorch inside the foreground Mimics process.

The annotator workflow is:

1. Import cases into `.mcs`.
2. Annotate a small number of saved cases.
3. Select a Mask named as the target organ.
4. Start background training.
5. Open another saved `.mcs`, select the same organ Mask, and run prediction.
6. Choose whether to update the selected Mask or create a new editable result,
   then review and refine it manually or with nnInteractive.

## Annotator Experience

The Mimics Scripting Library exposes five ordered actions:

- `01 Train Model`
- `Predict Current Case`
- `Predict Choose Model`
- `Show Status Results`
- `Stop AI Task`

The annotator should not need to remember paths or training parameters during ordinary use. When a project is opened from `<dataset>/mcs_output/<case>.mcs`, the integration infers both the dataset root and current case from that path. If that inference fails, the user is asked to select the dataset folder.

The training entry uses the active Mask name as the organ/task and opens one external setup window. Mimics writes a small context JSON, starts `tools/fewshot_training_setup_ui.py` with `Popen`, and returns immediately; it does not wait for parameter input. The window has two task-oriented pages: `Data and samples` and `Model and policy`. There is no separate basic/advanced workflow. When the user clicks Start Training, the external process starts `tools/fewshot_pipeline.py train` and writes the normal training job status file. Mimics only polls status files and logs progress.

Entry behavior:

| Entry | Main use | Foreground Mimics work |
| --- | --- | --- |
| `Train Model` | Choose samples and strategy, then train the selected organ. | Start external setup UI and monitor its status JSON. |
| `Predict Current Case` | Apply the latest model for the selected organ to the currently open case. | Start external inference, poll status, apply final mask buffer. |
| `Predict With Model...` | Choose a specific local/global model version before inference. | Show model chooser only when available, then same as prediction. |
| `Show Status` | Inspect jobs, logs, training curve, resource waits and results. | Start external status viewer; text dialog only as fallback. |
| `Stop AI Task` | Cancel the active DINOv3 task for the selected dataset. | Write a cancel marker; let the controller release checkpoints and GPU, with targeted termination only if its controller is already gone. |

The external setup and status windows use PySide6 from `nninteractive_env`, not the embedded Mimics Python session. They are not topmost, do not call back into Mimics, and do not wait on Mimics APIs. Tkinter remains only a functional fallback when PySide6 cannot load. If neither external backend is available, the setup status records a clear dependency error. No dataset export, image loading, training, or inference runs in the foreground Mimics process.

Training starts only after a reminder that saved `.mcs` files are used. This prevents a common failure mode where the annotator has edited the current case but has not saved it yet, so the background export would train from old labels.

To avoid GPU and memory contention, the Mimics entry does not start a second few-shot task while another training or inference task is still active for the same dataset. A cross-subsystem GPU lock also prevents DINOv3 training/inference from running at the same time as the managed nnInteractive server. The user can inspect waiting/running work with `04 Show Status Results` or stop it with `05 Stop AI Task`.

## Progress And Results

Every job writes a JSON status file under:

```text
<dataset>/fewshot_models/jobs/
```

`Show Status` opens an external read-only status window by default. Mimics only
starts `tools/fewshot_status_viewer.py` and returns; the status window polls
JSON/log files outside the Mimics GUI process. If the external status viewer
cannot be launched and `status_ui_fallback_to_text` is enabled, Mimics falls
back to the previous non-blocking text status dialog.

The status window shows:

- only the active task for the selected organ Mask, or its most recent task when none is active;
- current activity details: organ, case, status, strategy, sample counts,
  resource wait, latest progress, result state, and actionable error;
- a lightweight training curve parsed from epoch logs: train loss and
  validation Dice;
- a separate Technical details page containing IDs, paths, warnings, and bounded log tails;
- `Open Workspace`, `Open Log Folder`, and `Request Stop`.

`Request Stop` uses only the cancel marker and PIDs recorded in the selected job
JSON. It does not scan by broad process names and does not terminate the
foreground Mimics process.

The text fallback displays the same single selected task with:

- job id;
- job type;
- organ;
- current state;
- setup state such as `Configuring training` while the external training window is open;
- resource wait reason when the job is waiting for GPU or background Mimics;
- epoch progress when training has started;
- train/validation sample counts and selected case IDs;
- sample count;
- best Dice reported by the trainer when available;
- latest loss and validation Dice when available;
- latest log path;
- completed model checkpoint path;
- prediction output path.

The DINOv3 trainer writes structured training progress after initialization, during train/validation batches, and after each epoch. The Mimics monitor reads this JSON status and logs visible training milestones with epoch, batch, phase, train loss, validation Dice, learning rate and best validation Dice. Text logs still exist, but the Mimics UI does not depend on parsing text logs.

The external training setup window also keeps a Status panel open after training starts. It polls the same task JSON approximately every 1.5 seconds and shows the current stage, sample counts, epoch, loss, validation Dice when validation is enabled, best Dice, errors and completion state. Closing the external setup window does not stop training; `05 Stop AI Task` or the status viewer `Request Stop` remain the cancellation paths.

The job JSON keeps stable machine-readable states such as `waiting_for_gpu` and
`waiting_for_background_mimics`. Mimics renders those as user-facing phrases
such as "Waiting for GPU" and "Waiting for background Mimics" so the annotator
can tell that the job is queued for a resource rather than frozen.

Completed models are registered under:

```text
<dataset>/fewshot_models/models/<organ>/latest.json
```

The latest registered model is used by default for prediction. Older model versions stay available and can be selected with `Predict With Model...`.

## Stopping Work

`05_Stop_AI_Task` writes the task cancel marker and changes the state to
`cancelling`. The controller gives training up to 30 seconds to finish the
current step, flush status/checkpoints, release CUDA memory, and remove the GPU
lock. If the controller is already gone, only the recorded worker PID is
terminated. The final controller state is `cancelled`.

The trainer also checks the cancel marker during training. If it sees the marker before the forced termination arrives, it exits cleanly and reports `cancelled`.

Global cleanup through `03_Stop_All_Owned_Services.py` remains available for exceptional cases, but normal stopping should use `05_Stop_AI_Task` first because it updates the job state.

## Non-Blocking Rule

Mimics must not train, infer, scan large datasets, or convert large image files in the foreground GUI process.

Foreground Mimics only performs:

- lightweight menu selection;
- background process launch;
- status polling through `QTimer` or Win32 timer;
- final `set_voxel_buffer` when a completed prediction is applied.

All long work runs outside the foreground Mimics process:

- saved `.mcs` export runs in background Mimics;
- dataset preparation runs in `tools/fewshot_pipeline.py`;
- DINOv3 training runs in the Python environment configured by `fewshot_config.json`;
- DINOv3 inference runs in the same external Python environment;
- NIfTI prediction to Mimics `.u8` buffer conversion runs through `mimics_bridge.py`.

## Resource Lifecycle

The integration uses file-backed resource locks under:

```text
<repo>/.mimics_runtime/locks/
```

`gpu.lock` is shared by DINOv3 and the managed nnInteractive server. Training defaults to waiting up to 24 hours, inference defaults to waiting up to 1 hour, and nnInteractive defaults to a short 30 second wait. A waiting DINOv3 job reports `waiting_for_gpu` and records the lock holder in its job JSON instead of competing for CUDA memory and risking OOM.

`background_mimics.lock` is shared by import `.mcs` creation, label export, and few-shot label export. Only one Mimics-Script-owned `MimicsResearch.exe -b` process should run at a time. This is intentionally conservative because floating-license capacity is not visible to the scripts.

Default startup cleanup is safe by default: it removes stale resource lock files whose PID no longer exists, and nnInteractive can clean an owned server whose watchdog has disappeared after the idle timeout. It does not kill live bridge, training, inference, or background Mimics processes unless the user explicitly runs `Stop Background Services` or sets `MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START=1`.

## Organ Selection

The organ is intentionally not typed by the user.

The Mimics entry reads the currently selected Mask name and uses it as the organ key. This keeps the annotator action cheap and avoids text input errors.

Recommended convention:

- manual annotation Mask: `liver`, `kidney_left`, `tumor`, etc.;
- optional editable AI result Mask: `<source name> - AI Draft`.

Training and inference use the selected manual organ name. Inference starts
without an output-target dialog. When the converted result is ready and the
original Mimics case/grid has been verified, one `Prediction Ready` dialog lets
the annotator update the originally selected Mask or create a unique
`<source name> - AI Draft` Mask. That choice is also the completion notification, so no second
success dialog is shown. If the selected Mask changes while inference is
running, manual edits are preserved and the result is redirected to a new Mask.

## Sample Selection

The default sample policy is `all labeled`.

For organ `liver`, a case becomes a training sample when both files exist:

- image: `<dataset>/<case>/ct.nii.gz`, `mri.nii.gz`, or another top-level NIfTI;
- label: `<dataset>/<case>/segmentations/liver.nii.gz`.

Default training uses all eligible cases for the selected organ. Advanced training can restrict the case list, cap samples with `max_samples`, choose `latest` label order, and set a validation fraction. If exact cases are selected in the advanced window, only those cases are exported and used.

The external CLI exposes the same behavior:

```bash
python tools/fewshot_pipeline.py discover --ts-root /path/to/dataset --organ liver
python tools/fewshot_pipeline.py train --ts-root /path/to/dataset --organ liver --cases s0001,s0004,s0010
```

Saved `.mcs` files are exported first through a background Mimics process, so new annotations become available as NIfTI labels without freezing the open Mimics GUI.

For training, each Mask is transformed from the actual Mimics image grid back
to the original source-image NIfTI grid using physical coordinates and
nearest-neighbor resampling. A fresh export must match the source image in both
shape and affine. The pipeline stops if it does not; it no longer hides an
export error by resampling the source image onto the Mask grid. Consequently,
training and inference both start from the same original image orientation.

## Model Workspace

Each dataset root gets an isolated workspace:

```text
<dataset>/fewshot_models/
  jobs/
  datasets/<organ>/<run_id>/
  runs/<organ>/<run_id>/
  models/<organ>/<run_id>/
  models/<organ>/latest.json
  predictions/<case>/<organ>/
  fewshot_pipeline.log
```

Model registration stores:

- model id;
- organ name;
- copied checkpoint path;
- copied config path;
- config SHA-256 and the effective strategy/configuration;
- source DINOv3 experiment path;
- sample list;
- sample count;
- DINOv3 project path;
- base config path.

Inference uses `latest.json` by default. Older model versions remain under `models/<organ>/<run_id>/`. The Mimics `Predict With Model...` entry can select either a local dataset model or a reusable model from the user-level global registry.

Training also registers a compact global index at:

```text
~/.mimics_script/fewshot_model_index.json
```

The global index stores only model metadata and manifest paths. It does not copy checkpoints and it does not overwrite previous model versions. This allows a later dataset with the same organ name to discover an existing local model without forcing the annotator to remember where it was trained.

Materialized training datasets under `fewshot_models/datasets/<organ>/<run_id>/` are deleted after successful model registration by default because they duplicate the source image/label files. Failed runs keep the materialized data for debugging. Advanced training can opt into keeping the materialized dataset.

## Training Parameters

The Mimics UI exposes one `01_Train_Model.py` entry. The external setup window
contains two pages in the same workflow:

- `Data and samples`: dataset root, explicit case selection, sample cap, and
  validation split;
- `Model and policy`: data policy, model adaptation,
  decoder, installed model scale, image size, epochs, gradient accumulation,
  learning rate and schedule, warmup, weight decay, validation fraction, and mixed precision.

Batch size is fixed at one because case depth varies. Gradient accumulation is
the supported effective-batch control. Sub-volume depth, checkpoint retention,
modality, base config, and custom weight paths are backend settings and are not
shown in the normal annotator window.

The interactive UI prototype is stored at
`docs/previews/dinov3_fewshot_interactive.html`. It can be opened directly and
demonstrates setup, dependency rules, current-task status, progress, and logs.

Only parameters wired into the current pipeline and trainer are exposed:

- fine-tuning method, LoRA rank/alpha, adapter bottleneck, decoder, image
  detail, learning rate, weight decay, epoch count, gradient accumulation, and
  mixed precision are written into the generated DINOv3 YAML;
- pretrained model scale maps to an installed local model directory under `external/dinov3-medical-seg/models`; unavailable scales are not advertised in the external UI;
- image detail is a labeled preset over common `data.img_size` values
  (`192,192`, `224,224`, `256,256`, `320,320`) plus an editable `Custom size`;
- `No validation` writes `training.validation_enabled: false`, so the DINOv3 trainer does not reuse training data as validation.

The preset selector provides three conservative starting points and is not an
organ lookup table. Users may change the supported fields after applying a
preset; the complete resolved policy is stored with the model.

| Starting preset | Purpose | Initial configuration |
| --- | --- | --- |
| Adaptive | Conservative default for a new task | Fingerprint chooses full-volume or patch sampling |
| Full-volume baseline | Targets sufficiently represented in the field of view | Full sampling, Dice-Focal, whole inference |
| Patch | Small or sparse targets | Fingerprint-sized patches and sliding-window inference |

Exposed policy controls include sampling mode, patch sizing/focus, slice plane,
single-slice versus 2.5D input, neighbor distance, loss family, and optional
largest-component post-processing. Focal coefficients remain at the values used
by the controlled experiments; per-class weights are computed from training
labels. CT uses the configured modality-aware CT normalization, while MRI uses
robust percentile normalization, so a CT window is not shown as a generic user
control.

Dependencies are enforced twice, in the UI and in the background compiler:

- full-volume sampling forces whole-volume inference and disables patch fields;
- patch sampling automatically uses sliding-window inference;
- custom patch size is enabled only in custom patch-size mode;
- neighbor distance is enabled only for 2.5D input;
- LoRA and adapter controls are mutually enabled by the selected adaptation method.

Sampling dimensionality, encoder context, and decoder dimensionality are
independent controls. Full and patch sampling always select a 3D volume or 3D
sub-volume. A 2D decoder processes every selected slice and stacks predictions
back into a 3D mask; a 3D decoder fuses the feature volume. The 2.5D channel
policy supplies neighboring-slice context to the encoder and is valid with
either decoder family. The normal UI exposes `conv2d` as the only 2D decoder
with a completed pilot result; unconfirmed 2D variants remain research-only.

Invalid combinations passed through the CLI are rejected before label export,
GPU allocation, or training. The same generated YAML is copied into the model
registry, verified by SHA-256, and reused by inference, so custom preprocessing
and prediction policy cannot be lost between training and Mimics application.

Runtime UI startup does not import or require Pillow or a browser. The
interactive HTML is documentation only; the deployed external window remains
PySide6 and uses the same policy field names and dependency rules.

All DINOv3 Scripting Library entries are thin wrappers through
`runtime_py35/_mimics_entrypoint.py`. They do not reload
`fewshot_mimics.py` on every click, so active prediction monitors are not reset
when the annotator opens status, starts another allowed action, or stops a job.

Defaults live in `fewshot_config.json`:

- bundled DINOv3 project path, normally `external/dinov3-medical-seg`;
- external Python path;
- base config;
- validated base config `config/research/ct_fewshot_fast.yaml` for a 12 GB GPU;
- UI mode: `advanced_ui_mode` defaults to `external`;
- fallback behavior: `advanced_ui_fallback_to_internal` defaults to enabled so a missing external UI does not silently use defaults;
- fine-tuning method: `frozen`/decoder-only, `lora`, `adapter`, or `full`;
- decoder: `linear3d`, `mlp_probe`, `segformer3d`, or `dpt3d`;
- pretrained model scale: `vitb16`, `vitl16`, or `vith16plus` when the corresponding local model directory exists;
- epochs;
- batch size, fixed to one in the annotator workflow;
- gradient accumulation;
- learning rate and weight decay;
- image size;
- modality;
- validation fraction;
- minimum and maximum samples;
- mixed precision and backend-only sub-volume options;
- checkpoint retention: `best_model.pth` plus the last N epoch checkpoints.

Rationale:

- annotators should choose organ and action, not tune hyperparameters;
- project owners can adjust defaults centrally;
- advanced runs can use the external CLI.

## Inference And Apply

Prediction flow:

1. User opens `<dataset>/mcs_output/<case>.mcs`.
2. User selects the organ Mask.
3. Mimics starts `fewshot_pipeline.py infer`.
4. External Python writes a prediction NIfTI.
5. Mimics detects completion through a timer.
6. `mimics_bridge.py mask_to_buffer` converts the NIfTI result to the launch-time Mimics image grid and buffer order.
7. Mimics updates the launch-time selected Mask or creates a unique
   `<source name> - AI Draft`, according to the choice made when the result is ready.

`Predict Current Case` uses the latest local model for the active Mask name. `Predict With Model...` lets the annotator select a specific run or reusable model before inference.

The current case is inferred from the open project path. If the project is not under `<dataset>/mcs_output/<case>.mcs`, prediction is not started because the workflow cannot safely match the open image to the dataset case.

Before launch, Mimics measures the active image grid from
`get_voxel_center()` and records its shape and voxel-to-RAS matrix. The source
NIfTI path, shape, and affine must also match the imported project metadata.
Inference uses the exact copied training YAML; its SHA-256 is verified before
the GPU starts. Training and inference both canonicalize to RAS internally,
apply the same intensity/channel/image-size pipeline, and restore the result to
the original input NIfTI grid.

If the annotator opens another `.mcs` while inference runs, the completed
prediction remains in `waiting_for_source_case`. It is not applied to the new
case. Application resumes only after the original project and the same physical
image grid are open again. There is no source-geometry fallback when the live
Mimics grid cannot be verified.

## Failure Handling

Each training or inference job writes a JSON status file under `fewshot_models/jobs/`.

Common failure cases:

- no selected organ Mask;
- current project path cannot be matched to a dataset case;
- not enough exported labels for the organ;
- external Python or DINOv3 project not found;
- DINOv3 training or inference exits non-zero;
- prediction conversion fails because geometry does not match the case image.
- a fresh exported label does not match the source-image grid;
- the registered training config changed after model registration;
- the original project is not open when a completed prediction is ready.

The foreground Mimics process reports failures through non-blocking dialogs and leaves detailed logs in the workspace.

## Background Cleanup

`03_Stop_All_Owned_Services.py` and `tools/mimics_batch_cli.py kill-background` include `fewshot_pipeline.py` in their process markers.

The cleanup intentionally does not match generic `scripts/train.py` or `scripts/infer.py`, because those names may also be used by unrelated experiments. If a child training process survives after a forced kill, use the PID in the job JSON for precise cleanup.

## Current Limitations

- DICOM-only cases are not used for DINOv3 training or inference unless a NIfTI image is also present.
- Validation uses an explicit held-out split when enough selected samples are available. If the selected sample count is too small, validation is skipped or falls back to available materialized data, so Dice should still be interpreted as workflow feedback rather than an unbiased benchmark.
- Applying a prediction still calls `set_voxel_buffer` in the foreground Mimics process. This is much smaller than training/inference but can still take a short moment for very large volumes.
- Real Mimics validation is still required for vendor/version-specific behavior of `get_voxel_center()` and the final foreground `set_voxel_buffer()` call.

## External CLI Examples

Train or update an organ model:

```bash
python tools/fewshot_pipeline.py train \
  --ts-root /path/to/dataset \
  --organ liver \
  --cases s0001,s0004,s0010 \
  --finetune-method lora \
  --decoder segformer3d \
  --model-scale vitb16 \
  --epochs 10 \
  --batch-size 1 \
  --val-fraction 0.2 \
  --export-labels
```

Run inference for one case:

```bash
python tools/fewshot_pipeline.py infer \
  --ts-root /path/to/dataset \
  --case-id s0003 \
  --organ liver \
  --model-id latest
```

Run inference with a reusable model manifest:

```bash
python tools/fewshot_pipeline.py infer \
  --ts-root /path/to/new_dataset \
  --case-id s0003 \
  --organ liver \
  --model-manifest /path/to/fewshot_models/models/liver/train_x/manifest.json
```

List registered latest models:

```bash
python tools/fewshot_pipeline.py list-models --ts-root /path/to/dataset --all --include-global
```

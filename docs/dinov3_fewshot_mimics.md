# DINOv3 Few-Shot Mimics Integration Design

## Goals

This integration adds a few-shot organ segmentation workflow to Mimics without running PyTorch inside the foreground Mimics process.

The annotator workflow is:

1. Import cases into `.mcs`.
2. Annotate a small number of saved cases.
3. Select a Mask named as the target organ.
4. Start background training.
5. Open another saved `.mcs`, select the same organ Mask, and run prediction.
6. Review the generated `AI_<organ>` Mask and edit it as needed.

## Annotator Experience

The Mimics entry exposes four actions:

- `Train/Update Model`
- `Predict Current Case`
- `Show Status`
- `Stop Latest Job`

The annotator should not need to remember paths or training parameters during ordinary use. When a project is opened from `<dataset>/mcs_output/<case>.mcs`, the integration infers both the dataset root and current case from that path. If that inference fails, the user is asked to select the dataset folder.

The ordinary training entry uses the active Mask name as the organ/task and runs the configured default training profile. The advanced training entry opens one lightweight settings window where the annotator can choose cases, validation split, fine-tuning method, decoder, model scale, image size, epochs, batch size, gradient accumulation and learning rate. If Qt bindings are not available in the embedded Mimics Python session, the entry falls back to a profile selector backed by `fewshot_config.json` and a small native Mimics parameter picker instead of silently using defaults. The Mimics process does not recursively scan Program Files for Qt bindings. If a Qt dialog is required, set `MIMICS_QT_PYTHONPATH` or `MIMICS_PYQT_PATH` to a Python-3.5-compatible PyQt5/PySide site-packages path. Do not install a modern PyQt5 wheel into Mimics Python 3.5. No dataset export, image loading, training or inference runs in the Mimics foreground process.

Training starts only after a reminder that saved `.mcs` files are used. This prevents a common failure mode where the annotator has edited the current case but has not saved it yet, so the background export would train from old labels.

To avoid GPU and memory contention, the Mimics entry does not start a second few-shot job while another training or inference job is still active for the same dataset. A cross-subsystem GPU lock also prevents DINOv3 training/inference from running at the same time as the managed nnInteractive server. The user can inspect waiting/running work with `Show Status` or stop it with `Stop Latest Job`.

## Progress And Results

Every job writes a JSON status file under:

```text
<dataset>/fewshot_models/jobs/
```

`Show Status` displays the latest jobs with:

- job id;
- job type;
- organ;
- current state;
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

`Stop Latest Job` writes the job cancel marker and requests termination of the active pipeline/training/inference process tree. The job status is changed to `cancelled`.

The trainer also checks the cancel marker during training. If it sees the marker before the forced termination arrives, it exits cleanly and reports `cancelled`.

Global cleanup through `Stop_Background_Services.py` remains available for exceptional cases, but normal stopping should use `Stop Latest Job` first because it updates the job state.

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
- AI result Mask: `AI_liver`, `AI_kidney_left`, `AI_tumor`.

Training and inference use the selected manual organ name. The prediction result is written to `AI_<organ>` so manual labels are not overwritten.

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

The Mimics UI exposes two levels:

- `DINOv3_Train_Update_Model.py`: default profile, minimal prompts.
- `DINOv3_Train_Advanced.py`: one settings window for sample and parameter selection when PyQt5 is available; otherwise a non-PyQt profile selector for the profiles in `fewshot_config.json`.

All DINOv3 Scripting Library entries are thin wrappers through
`scripting_library/_mimics_entrypoint.py`. They do not reload
`fewshot_mimics.py` on every click, so active prediction monitors are not reset
when the annotator opens status, starts another allowed action, or stops a job.

Defaults and profiles live in `fewshot_config.json`:

- bundled DINOv3 project path, normally `external/dinov3-medical-seg`;
- external Python path;
- base config;
- training profiles such as `balanced`, `fast_check`, `low_memory`, and `quality_lora`;
- fine-tuning method: `frozen`/decoder-only, `lora`, `adapter`, or `full`;
- decoder: `linear3d`, `mlp_probe`, `segformer3d`, or `dpt3d`;
- pretrained model scale: `vitb16`, `vitl16`, or `vith16plus`;
- epochs;
- batch size;
- gradient accumulation;
- learning rate and weight decay;
- image size;
- modality;
- validation fraction;
- minimum and maximum samples;
- mixed precision and sub-volume options.
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
6. `mimics_bridge.py mask_to_buffer` converts the NIfTI result to the current active Mimics image grid and buffer order.
7. Mimics creates or updates `AI_<organ>`.

`Predict Current Case` uses the latest local model for the active Mask name. `Predict With Model...` lets the annotator select a specific run or reusable model before inference.

The current case is inferred from the open project path. If the project is not under `<dataset>/mcs_output/<case>.mcs`, prediction is not started because the workflow cannot safely match the open image to the dataset case.

When applying a prediction to an open `.mcs`, the conversion bridge first tries
to use `mimics_script.mimics_voxel_to_ras_matrix` stored on the active image.
That matrix is derived from the actual imported Mimics image grid, so prediction
masks are resampled into the same grid that `Mask.set_voxel_buffer()` expects.
If that metadata is unavailable, the bridge falls back to the source image
geometry and logs a warning.

## Failure Handling

Each training or inference job writes a JSON status file under `fewshot_models/jobs/`.

Common failure cases:

- no selected organ Mask;
- current project path cannot be matched to a dataset case;
- not enough exported labels for the organ;
- external Python or DINOv3 project not found;
- DINOv3 training or inference exits non-zero;
- prediction conversion fails because geometry does not match the case image.

The foreground Mimics process reports failures through non-blocking dialogs and leaves detailed logs in the workspace.

## Background Cleanup

`Stop_Background_Services.py` and `tools/mimics_batch_cli.py kill-background` include `fewshot_pipeline.py` in their process markers.

The cleanup intentionally does not match generic `scripts/train.py` or `scripts/infer.py`, because those names may also be used by unrelated experiments. If a child training process survives after a forced kill, use the PID in the job JSON for precise cleanup.

## Current Limitations

- DICOM-only cases are not used for DINOv3 training or inference unless a NIfTI image is also present.
- Validation uses an explicit held-out split when enough selected samples are available. If the selected sample count is too small, validation is skipped or falls back to available materialized data, so Dice should still be interpreted as workflow feedback rather than an unbiased benchmark.
- Applying a prediction still calls `set_voxel_buffer` in the foreground Mimics process. This is much smaller than training/inference but can still take a short moment for very large volumes.
- The default Mimics UI stays simple; advanced controls are available from a separate entry to avoid adding setup burden to routine annotation.

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

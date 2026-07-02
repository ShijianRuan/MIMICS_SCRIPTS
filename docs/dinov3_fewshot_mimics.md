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

Training starts only after a reminder that saved `.mcs` files are used. This prevents a common failure mode where the annotator has edited the current case but has not saved it yet, so the background export would train from old labels.

To avoid GPU and memory contention, the Mimics entry does not start a second few-shot job while another training or inference job is still active for the same dataset. The user can inspect it with `Show Status` or stop it with `Stop Latest Job`.

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
- epoch progress when training has started;
- sample count;
- best Dice reported by the trainer when available;
- latest log path;
- completed model checkpoint path;
- prediction output path.

The DINOv3 trainer writes structured training progress after initialization and after each epoch. Text logs still exist, but the Mimics UI does not depend on parsing text logs.

Completed models are registered under:

```text
<dataset>/fewshot_models/models/<organ>/latest.json
```

The latest registered model is used by default for prediction.

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

The training command can also cap samples with `--max-samples` and choose the latest labels with `--sample-mode latest`, but these are not exposed in the Mimics UI by default. The UI should stay simple unless annotators repeatedly need those controls.

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

Inference uses `latest.json` by default. Older model versions remain under `models/<organ>/<run_id>/` and can be used externally through `tools/fewshot_pipeline.py infer --model-id <run_id>`.

## Training Parameters

The Mimics UI exposes no low-level training parameters by default.

Defaults live in `fewshot_config.json`:

- bundled DINOv3 project path, normally `external/dinov3-medical-seg`;
- external Python path;
- base config;
- epochs;
- batch size;
- gradient accumulation;
- learning rate;
- image size;
- modality;
- minimum and maximum samples.

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
6. `mimics_bridge.py mask_to_buffer` converts the NIfTI result to Mimics buffer order.
7. Mimics creates or updates `AI_<organ>`.

The current case is inferred from the open project path. If the project is not under `<dataset>/mcs_output/<case>.mcs`, prediction is not started because the workflow cannot safely match the open image to the dataset case.

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
- Validation currently reuses available labeled cases if no independent test split exists, so Dice metrics are useful for smoke testing but not for unbiased model selection.
- Applying a prediction still calls `set_voxel_buffer` in the foreground Mimics process. This is much smaller than training/inference but can still take a short moment for very large volumes.
- The first implementation keeps the Mimics UI simple. Advanced sample policy and training parameters are available through the external CLI rather than extra Mimics dialogs.

## External CLI Examples

Train or update an organ model:

```bash
python tools/fewshot_pipeline.py train \
  --ts-root /path/to/dataset \
  --organ liver \
  --export-labels
```

Run inference for one case:

```bash
python tools/fewshot_pipeline.py infer \
  --ts-root /path/to/dataset \
  --case-id s0003 \
  --organ liver
```

List registered latest models:

```bash
python tools/fewshot_pipeline.py list-models --ts-root /path/to/dataset
```

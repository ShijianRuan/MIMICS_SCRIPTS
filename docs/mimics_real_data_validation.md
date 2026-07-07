# Mimics Real-Data Validation Checklist

This document lists the items that cannot be fully validated on a machine
without Mimics. Use it as the shared checklist when testing on a Windows Mimics
workstation with real data. Please keep log messages and copied error text in
English when reporting results, because the integration logs are standardized in
English.

## Before Testing

Use the latest pushed commit:

```powershell
git pull origin main
git rev-parse --short HEAD
```

Record the commit hash in the report. If background processes may still be
running from previous tests, stop only integration-created processes:

```powershell
python tools\mimics_batch_cli.py kill-background
```

Inside Mimics, the equivalent entry is:

```text
Scripting Library > 99_Admin > Stop_Background_Services
```

Useful log locations:

- Import logs: `<ts_root>\mcs_output\logs\mimics_import.log`
- Import background Mimics log: `<ts_root>\mcs_output\_background_mimics.log`
- Import per-case prepare manifest: `<ts_root>\mcs_output\<case_id>_work\prepare_manifest.json`
- Import failures: `<ts_root>\mcs_output\_failed_cases\*.json`
- Export logs: `<ts_root>\mcs_output\mimics_export.log`
- Export background Mimics log: `<ts_root>\mcs_output\_background_export_mimics.log`
- nnInteractive logs: `<repo>\.mimics_runtime\nninteractive\logs\nninteractive_mimics.log`
- nnInteractive bridge/server logs: paths shown in Mimics error dialogs, usually under `<repo>\.mimics_runtime\nninteractive\`
- DINOv3 log: `<ts_root>\fewshot_models\fewshot_pipeline.log`
- DINOv3 job status: `<ts_root>\fewshot_models\runs\<organ>\<run_id>\job_status.json`
- DINOv3 train progress: `<ts_root>\fewshot_models\runs\<organ>\<run_id>\train_status.json`
- Stop report: `<repo>\.mimics_runtime\stop_background_last.json`

## Test Data Needed

Prepare a small but representative validation set:

- 2 TotalSegmentator-style NIfTI cases with `ct.nii.gz` and `segmentations\*.nii.gz`.
- 1 case where the mask affine differs from the image affine but should still
  physically overlap after resampling.
- 1 original DICOM folder case, preferably with non-trivial orientation or slice
  ordering.
- Optional but valuable: MR DICOM or MR NIfTI, oblique CT, and one case on a
  network drive if that is part of normal use.
- At least one already annotated `.mcs` with several named masks for DINOv3 and
  export validation.

Avoid sending patient-identifying data back. Screenshots can be cropped to the
image viewport and logs can be redacted for paths if needed.

## 1. Import And MCS Orientation

Goal: confirm that imported image and mask are aligned in Mimics for arbitrary
source orientations.

Run from inside Mimics:

```text
Scripting Library > 01_Data > Import_Dataset
```

Select the dataset root. For each converted `.mcs`:

- Open the generated `.mcs`.
- Activate the imported image.
- Show one organ mask that has a visually obvious boundary.
- Check axial, coronal and sagittal views.
- Confirm there is no left-right, anterior-posterior, superior-inferior, or
  diagonal mirror mismatch.
- Check both a mask whose affine matches the image and one whose affine differs.

External preparation path:

```powershell
python tools\mimics_batch_cli.py prepare-import --ts-root "D:\Dataset" --cases s0001,s0002 --python "D:\Mimics-Script\nninteractive_env\Scripts\python.exe" --mimics-exe "C:\Program Files\Materialise\Mimics Research 21.0\MimicsResearch.exe"
```

Prepare only, without creating `.mcs`:

```powershell
python tools\mimics_batch_cli.py prepare-import --ts-root "D:\Dataset" --cases s0001 --python "D:\Mimics-Script\nninteractive_env\Scripts\python.exe" --no-create-mcs
```

Report:

- Case id, source type, source image shape, source mask shape.
- Whether `prepare_manifest.json` contains `actual_mimics_grid_resampled: true`.
- Screenshots of all three views if any mismatch appears.
- `prepare_manifest.json`, `mimics_import.log`, `_background_mimics.log`, and
  the relevant `_failed_cases` JSON if a case failed.

Pass condition: masks overlap the image anatomy in all views, and failures are
explicitly logged instead of creating a wrong `.mcs`.

## 2. Batch Import Failure, Resume, And Disk Cleanup

Goal: confirm that failed cases do not corrupt the queue and successful cases do
not leave unnecessary derived DICOM/buffer data.

Recommended test:

1. Run a batch with 3-5 cases.
2. Include one intentionally broken case, for example a missing mask file or an
   unreadable image path.
3. Confirm successful cases still create `.mcs`.
4. Rerun the same command.
5. Confirm completed cases are skipped or reused correctly, and the failed case
   remains clearly reported.
6. Check disk usage under `<ts_root>\mcs_output`.

Report:

- `_mcs_queue_active.json`, `_mcs_queue_done.json`, and `_failed_cases\*.json`.
- Whether `<case_id>_work\derived_dicom` and buffers remain after successful
  `.mcs` creation.
- Any case where a failure blocked later cases.

Pass condition: one failed case does not stop unrelated cases, retry behavior is
understandable, and large intermediates are not retained for successful cases.

## 3. Import Responsiveness In Foreground Mimics

Goal: confirm that selecting a folder and starting discovery does not black-screen
or freeze the foreground Mimics GUI.

Test:

1. Open Mimics normally.
2. Run `Scripting Library > 01_Data > Import_Dataset`.
3. Select a dataset folder with many cases.
4. Immediately pan/zoom or interact with the existing open project.
5. Note the time between closing the folder dialog and seeing this log:

```text
Starting dataset discovery in the bridge process.
```

Report:

- Approximate freeze time in seconds.
- Whether the folder is local SSD, external disk, or network path.
- Whether antivirus or cloud sync is active on that folder.
- `mimics_import.log`.

Pass condition: any pause is short and clearly tied to the native folder dialog
return, not to Python-side dataset scanning.

## 4. nnInteractive Source Image Path

Goal: confirm that nnInteractive uses source image metadata when available and
does not silently fall back to the Mimics buffer.

Test:

1. Open an imported `.mcs`.
2. Select one target mask.
3. Run:

```text
Scripting Library > 02_AI > nnInteractive
```

4. Watch Mimics logging for:

```text
nnInteractive image input mode: source image. Mimics image buffer export is skipped.
```

5. Add one foreground point or lasso prompt.
6. Confirm the result applies automatically.
7. Record the timing summary:

```text
image_load / server_ready / set_image / set_target / prompt_apply / total
```

Negative test:

1. Temporarily move or rename the original source image file referenced by the
   `.mcs` metadata.
2. Run nnInteractive again.
3. Confirm it fails with an explicit missing source error instead of silently
   switching to Mimics buffer.

Report:

- The exact source-mode log line.
- Timing summary from Mimics logging.
- `nninteractive_mimics.log`, `nninteractive_bridge.jsonl`, and server log path
  if shown in a dialog.
- Whether the first prompt was slow because of `image_load`, `server_ready`,
  `set_image`, `set_target`, or `prompt_apply`.

Pass condition: source mode is used for imported projects, missing source paths
fail explicitly, and repeated prompts avoid unnecessary image reload work.

## 5. nnInteractive Prompt Workflows

Goal: confirm the integrated prompt behavior is usable and matches expected
nnInteractive workflow semantics.

Test the following on the same target mask:

- Existing non-empty mask as initial segmentation.
- Include and exclude point set.
- Foreground scribble and background scribble set.
- Box prompt.
- Lasso prompt on a single slice.
- Undo last prompt.
- Reset session.
- Edit the target mask while background inference is still running and confirm
  the stale result warning appears instead of overwriting newer manual work.

Report:

- Which prompt type caused a pause or black screen.
- Time from finishing prompt capture to this log:

```text
nnInteractive background inference started.
```

- Time from background start to result applied.
- Any empty prediction message and foreground voxel count.
- Worker status JSON if a worker stops before producing a result.

Pass condition: prompt capture may use Mimics dialogs/tools, but inference and
result waiting do not block the foreground GUI.

## 6. DINOv3 Few-Shot Training

Goal: confirm training integrates with the annotation workflow without blocking
Mimics and exposes enough progress to users.

Inside Mimics:

```text
Scripting Library > 02_AI > DINOv3 > Train_Update_Model
Scripting Library > 02_AI > DINOv3 > Train_Advanced
Scripting Library > 02_AI > DINOv3 > Show_Status
Scripting Library > 02_AI > DINOv3 > Stop_Latest_Job
```

External smoke test with few samples:

```powershell
python tools\fewshot_pipeline.py discover --ts-root "D:\Dataset" --organ liver --cases s0001,s0002
python tools\fewshot_pipeline.py train --ts-root "D:\Dataset" --organ liver --cases s0001,s0002 --epochs 3 --val-fraction 0.5 --run-id validation_liver
python tools\fewshot_pipeline.py list-models --ts-root "D:\Dataset" --organ liver --all
```

During training, verify `Show_Status` and `fewshot_pipeline.log` show epoch,
loss, learning rate, validation dice when validation is enabled, and best dice.

Cancel test:

```powershell
python tools\fewshot_pipeline.py cancel --ts-root "D:\Dataset"
```

Report:

- Training profile used, organ name, selected cases, train/validation counts.
- Whether `Show_Status` updates while training is still running.
- A few `Training progress:` log lines.
- Whether cancellation frees GPU memory and marks the job cancelled.
- `job_status.json`, `train_status.json`, `fewshot_pipeline.log`, and final
  `manifest.json` if training completed.

Pass condition: training runs outside foreground Mimics, progress is visible,
cancel works, and checkpoints are cleaned according to `keep_last_checkpoints`.

## 7. DINOv3 Inference And Mask Application

Goal: confirm DINOv3 predictions are converted into the active Mimics grid before
being applied to masks.

Inside Mimics:

```text
Scripting Library > 02_AI > DINOv3 > Predict_Current_Case
Scripting Library > 02_AI > DINOv3 > Predict_Choose_Model
```

External inference:

```powershell
python tools\fewshot_pipeline.py infer --ts-root "D:\Dataset" --case-id s0003 --organ liver --model-id latest
```

Report:

- Whether the prediction mask appears on the correct anatomy in all three views.
- Whether Mimics logs say:

```text
DINOv3 prediction conversion will use the active Mimics image grid
```

- If it falls back to source image geometry, include the active project metadata
  situation and logs.

Pass condition: DINOv3 output mask aligns with the current `.mcs` image and does
not require manual flip/rotate fixes.

## 8. GPU And Background Mimics Resource Contention

Goal: confirm background tasks queue rather than crashing each other.

Test:

1. Start an nnInteractive inference that keeps the server warm.
2. Start a DINOv3 training job.
3. Confirm DINOv3 waits for the GPU lock or reports a clear wait status.
4. Start batch `.mcs` creation and batch label export near the same time.
5. Confirm only one background Mimics task owns the `background_mimics.lock`.

Report:

- `.mimics_runtime\locks\gpu.lock`
- `.mimics_runtime\locks\background_mimics.lock`
- `fewshot_pipeline.log`
- Any Mimics license or background Mimics startup errors.

Pass condition: tasks wait, fail clearly, or can be cancelled; they do not cause
CUDA OOM, hidden stuck workers, or foreground Mimics instability.

## 9. Stop Background Services

Goal: confirm manual stop releases integration-owned processes without killing
foreground Mimics or unrelated processes.

Run inside Mimics:

```text
Scripting Library > 99_Admin > Stop_Background_Services
```

Or from PowerShell:

```powershell
python tools\mimics_batch_cli.py kill-background
```

Report:

- Whether foreground Mimics stayed open.
- `stop_background_last.json`.
- Whether GPU memory was released after stopping nnInteractive/DINOv3.
- Any process that remained unexpectedly.

Pass condition: only Mimics-Script-owned bridge, worker, server, watchdog, and
background Mimics processes are stopped.

## 10. Window/Level Presets

Goal: confirm preset values are correct in real Mimics GV range and reset/undo do
not crash.

Inside Mimics:

```text
Scripting Library > 03_Display > Window_From_Selected_Mask
Scripting Library > 03_Display > Window_Choose_Preset
Scripting Library > 03_Display > Window_Reset_Full_Range
Scripting Library > 03_Display > Window_Undo_Last
```

Test:

- Select a mask with a standardized organ name.
- Apply the automatic preset from the selected mask.
- Apply a manually chosen preset.
- Reset full range.
- Undo last window/level change.
- Repeat on CT cases with different min/max GV ranges.

Report:

- Image minimum and maximum if shown in logs.
- Exact error if Reset Full Range fails.
- Whether the contrast visually matches the intended organ.

Pass condition: presets clamp to the Mimics-accepted range, Reset Full Range does
not raise `ValueError`, and Undo restores the previous contrast.

## 11. External Batch Export

Goal: confirm labels can be exported without using the foreground Mimics window.

Run:

```powershell
python tools\mimics_batch_cli.py export-labels --ts-root "D:\Dataset" --cases s0001,s0002 --mimics-exe "C:\Program Files\Materialise\Mimics Research 21.0\MimicsResearch.exe"
```

Report:

- Exported label paths and names.
- New/overwritten/unchanged counts from `mimics_export.log`.
- Any missing `.mcs` or missing mask reports.

Pass condition: export runs in background Mimics, foreground Mimics remains
usable, and exported label orientation matches the original image/mask geometry.

## Report Template

Copy this block when sending results back:

```text
Commit:
Mimics version:
Windows version:
GPU / driver / CUDA:
Python env used for bridge:
Dataset type:
Case ids:

Test section:
Command or Scripting Library entry:
Expected:
Actual:
Foreground Mimics responsive? yes/no, freeze time:
Image/mask alignment in axial/coronal/sagittal:
Timing summary if nnInteractive:
Training progress lines if DINOv3:
Stop/cancel behavior:

Files attached or copied:
- mimics_import.log:
- _background_mimics.log:
- prepare_manifest.json:
- nninteractive_mimics.log:
- nninteractive_bridge.jsonl:
- fewshot_pipeline.log:
- job_status.json:
- train_status.json:
- stop_background_last.json:

Screenshots:
Notes:
```

## Priority If Time Is Limited

Run these first:

1. Import two TotalSegmentator cases and visually confirm mask/image alignment in
   three views.
2. Run nnInteractive on one imported `.mcs` and confirm source image mode plus
   timing logs.
3. Train a 2-3 epoch DINOv3 smoke model and confirm progress/cancel/status.
4. Run DINOv3 inference and confirm prediction mask aligns in Mimics.
5. Run Stop Background Services and confirm foreground Mimics is not killed.

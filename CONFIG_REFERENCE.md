# Mimics-Script Configuration Reference

This document lists the supported user-facing configuration files and
environment variables. Paths may be absolute or relative to the project root
unless noted otherwise.

## `mimics_io_config.json`

| Key | Default | Purpose |
| --- | --- | --- |
| `mimics_output_dir` | `""` | Default folder for generated `.mcs` projects. An empty value uses `<dataset>/mcs_output`. This setting does not redirect mask exports, training runs, or logs. |
| `mimics_buffer_axes` | `[0, 1, 2]` | Optional advanced mapping from external array axes to the Mimics voxel buffer. Change only after real-data orientation validation. |
| `mimics_buffer_flips` | `[false, false, false]` | Optional advanced flips paired with `mimics_buffer_axes`. Change only after real-data orientation validation. |

## `interactive_algorithms_config.json`

Controls the interactive segmentation algorithms run outside Mimics
(ScribblePrompt; shared job retention for all algorithm jobs).

| Key | Default | Purpose |
| --- | --- | --- |
| `poll_seconds` | `0.25` | Mimics-side job monitor poll interval. |
| `job_retention_days` | `3` | Terminal job folder retention before cleanup. |
| `job_max_terminal` | `20` | Maximum terminal job folders kept. |
| `scribbleprompt.timeout_seconds` | `900` | Per-job external process timeout. |
| `scribbleprompt.device` | `"cpu"` | Inference device. |
| `scribbleprompt.input_size` | `128` | Model input resolution. |
| `scribbleprompt.prior_logit_magnitude` | `6.0` | Prior logit magnitude fed to the model. |
| `scribbleprompt.checkpoint` | `integrations/ScribblePrompt/checkpoints/...` | ScribblePrompt UNet checkpoint. |

## `window_level_presets.json`

CT window/level presets matched against Mask names (Review menu). Editable
from Mimics via **Review > Window Edit Presets**; no hand editing required.

Each entry: `name`, `width`, `level`, `keywords` (matched case-insensitively
against the selected Mask name, first keyword hit wins), and `source`
(citation for the W/L values). Ships with 5 presets: Lung, Abdomen / Soft
Tissue, Bone, Skull / Cranium, Vessel / Heart. State (undo/last applied)
lives in `ui_state/window_level_state.json`; user edits are saved back to
this file.

## Remote compute profiles (`servers.json`)

Server profiles for SSH/Docker remote training and batch inference
(FlexiCT / nnInteractive / nnU-Net). Managed by the connection setup UI
(training window > Compute tab); stored **outside the project** at
`%LOCALAPPDATA%\MimicsScript\remote_compute\servers.json` (override with
`MIMICS_REMOTE_CONFIG_DIR`). SSH host fingerprints live in `known_hosts`
next to it (first-connect confirmation, change = refuse). Passwords are
never stored in this file — they go to Windows Credential Manager
(`MimicsScript/remote/<profile-id>`), and key auth is preferred.

| Field | Purpose |
| --- | --- |
| `profile_id`, `name` | Stable id and display name. |
| `host`, `port`, `username` | SSH endpoint. |
| `auth_method` | `key` or `password`. |
| `runtime_image` | Container image for job containers (`--network none`, HF offline). |
| `gpu_device` | `auto` or an explicit device index. |
| `remote_root` | Work folder on the server. Relative paths resolve inside the SSH user's home directory. On a shared server, use the folder your administrator assigned rather than the account's own files. |
| `container_runtime` | `docker` (default) or `nerdctl` for containerd-only servers. |
| `container_namespace` | Optional nerdctl namespace for job containers; ignored by Docker. |
| `remote_code_verify` | Code-drift check on the shipped runtime: `strict` (refuse), `warn` (default), `off`. |
| `remote_weights_verify` | Downloaded-weights checksum check: `strict` (default), `warn`, `off`. |
| `remote_cache_retention_days` | Days finished jobs and cached training data stay on the server before cleanup (default 30, max 3650). |

All fields are editable in the connection setup UI (training window > Compute tab > Manage Servers); editing `servers.json` by hand is never required.

## `dataset_profiles.json`

Describes dataset folder layouts. Every discovery site (dataset import,
single-case import, mask export, the path-setup UI, and the bridge) reads this
one file, so changing the layout no longer means editing five places.

| Key | Default | Purpose |
| --- | --- | --- |
| `default_profile` | `"ts-like"` | Profile used when a caller does not name one. |
| `profiles.<id>.display_name` | profile id | Human-readable name. |
| `profiles.<id>.description` | `""` | Free-form note shown in tooling. |
| `profiles.<id>.image_candidates` | ts-like: `ct.nii.gz`, `mri.nii.gz`, ... | Preferred image file names, checked first, in order. Empty for the `generic` profile. |
| `profiles.<id>.fallback_image_suffixes` | `.nii.gz`, `.nrrd.gz`, `.nii`, `.mha`, `.mhd`, `.nrrd` | Suffixes accepted by the capped fallback scan when no preferred name matches. |
| `profiles.<id>.mask_dirs` | `["segmentations"]` | Sub-folders that hold label masks. Empty for `generic` (masks are only found next to the image). |
| `profiles.<id>.mask_suffixes` | `.seg.nii.gz`, `.seg.nii`, ... | Suffixes recognized as masks. |
| `profiles.<id>.dicom_dirs` | `["dicom"]` | Sub-folders treated as DICOM series. |
| `profiles.<id>.exclude_dirs` | `["mcs_output", "segmentations"]` | Child folders ignored when scanning a dataset root. Empty for `generic`. |

Built-in profiles:

- **`ts-like`** — byte-for-byte equivalent to the historical hard-coded
  behaviour. This equivalence is locked by
  `TestDatasetProfiles.test_fallback_matches_ts_like_byte_for_byte` and the
  discovery-equivalence test in `tools/test_all.py`.
- **`generic`** — no layout assumptions: any medical volume file in a case
  folder is the image, no folders are excluded, no preferred names exist.

If the file is missing or corrupt, every consumer silently falls back to a
built-in copy of the ts-like profile, so discovery never fails because the
config file was lost. Unknown profile ids fall back to the default profile
rather than raising.

The import path-setup UI additionally shows a one-line recognition summary
("Recognized N case(s); images: ... ; M mask(s)") refreshed in a background
thread, and interrupts once at submit when a case holds more than one volume
file or has no usable image.

## Import receipts (`.*.import_receipt.json`)

Every finished import writes a receipt next to the created `.mcs` file
(named `.<case>.import_receipt.json`, schema `mimics_import_receipt.v1`). It
records the masks the import created, the metadata keys it set, the project
path, and content fingerprints of the source image and of the saved `.mcs`.
Receipts are the basis of **Admin > Undo Last Import**
(`99_Admin/05_Undo_Last_Import.py`):

- masks listed in the receipt are deleted in one transaction;
- the `.mcs` file itself is deleted only when its fingerprint still matches
  the receipt (a true rollback); if the project changed since the import
  (annotations, extra masks), the file is kept and only the masks are
  removed;
- the receipt is consumed on success, so a second run cannot repeat the
  undo.

Receipt writing never fails an import: if the receipt cannot be written, a
warning is logged and the import still succeeds.

## Drop-to-import window state (`ui_state/io_paths.json`)

`01_Data/08_Quick_Drop_Import.py` opens an always-on-top external window
(`tools/import_drop_window.py`) that accepts dragged files, case folders,
dataset folders, or pasted paths, classifies them against the dataset
profile, and submits to the existing import workers. It is registered in the
process registry under role `external_ui` with cleanup policy
`idle_timeout_s:1800` and exits by itself after half an hour without input.
Its last-used output folder and mask selection are remembered in
`%LOCALAPPDATA%\Mimics-Script\ui_state\io_paths.json` under the
`drop_import` key (the same state file the path-setup UI uses).

## `fewshot_config.json` (removed)

The DINOv3 few-shot framework was removed. Its configuration keys
(`default_*` training defaults, `organ_mask_aliases`, external-UI switches,
`dinov3_project`) no longer exist. GPU, background-Mimics, and job-retention
settings now live in the pipeline-specific configs documented below.

## `nninteractive_config.json`

| Key | Default | Purpose |
| --- | --- | --- |
| `device` | `"auto"` | Inference device selection. |
| `image_input_mode` | `"mimics"` | Image source used by nnInteractive. |
| `task_model_image_input_mode` | `"auto"` | Input policy for custom task models: prefer a readable local source, otherwise use the portable image stored in the `.mcs`. MR rescale metadata/tags are restored when available; missing mappings continue as logged raw-GV best effort. Set `source` only for strict source-file parity, or `mimics` to always use the project buffer. |
| `prefer_source_image_for_nninteractive` | `false` | Prefer the original source image when enabled and geometry is validated. |
| `fallback_to_mimics_buffer_when_source_unavailable` | `true` | Use the Mimics image buffer if the source image cannot be validated. |
| `fallback_to_source_when_mimics_export_fails` | `false` | Permit the reverse fallback after a Mimics-buffer export failure. |
| `auto_start_server` | `true` | Start the external nnInteractive service on demand. |
| `async_start_worker_before_prompt` | `true` | Start image preparation before prompt collection. |
| `async_reuse_image_worker` | `true` | Reuse a prepared image worker within a session. |
| `incremental_interaction_replay` | `true` | Reuse prior prompt state when supported. |
| `existing_mask_result_mode` | `"ask"` | Ask whether to update the selected mask or create an editable copy. |
| `keep_server_warm_after_session` | `true` | Keep the server warm until its idle timeout. |
| `server_idle_timeout_seconds` | `300` | Warm-server idle lifetime. |
| `async_worker_idle_timeout_seconds` | `900` | Image-worker idle lifetime. |
| `server_startup_timeout_seconds` | `900` | Server startup deadline. |
| `set_image_timeout_seconds` | `1800` | Image upload/preprocessing deadline. |
| `prediction_timeout_seconds` | `1800` | Prediction deadline. |
| `bridge_timeout_seconds` | `4620` | Overall bridge deadline. |
| `gpu_lock_timeout_seconds` | `30` | GPU lock wait before reporting contention. |
| `minimum_free_gpu_memory_gb` | `4` | Free-VRAM floor checked before starting the CUDA server; below it the start fails with a "close other GPU programs" message instead of a mid-start OOM. |
| `async_poll_seconds` | `0.25` | Mimics-side async state polling interval. |
| `async_result_poll_seconds` | `0.25` | Result polling interval. |
| `async_job_retention_days` | `3` | Terminal async-job retention. |
| `async_job_max_terminal` | `20` | Maximum terminal async-job records. |
| `source_cache_retention_days` | `7` | Source-image cache retention. |
| `source_cache_max_entries` | `12` | Maximum source-image cache entries. |

## `nninteractive_finetune_config.json`

Controls the nnInteractive task fine-tuning pipeline (Model Center, label
export, training jobs, and quality gating).

| Key | Default | Purpose |
| --- | --- | --- |
| `workspace_dir` | `"nninteractive_task_models"` | Task-model workspace root (tasks, models, registry). |
| `official_model_dir` | `"nninteractive_env/models/nnInteractive_v1.0"` | Base model that fine-tuning starts from. |
| `default_strategy` | `"clopa_in"` | Default prompt-sampling strategy for new jobs. |
| `default_epochs` | `10` | Initial epoch spinner value in the training dialog. |
| `minimum_epochs` | `4` | Lower bound of the epoch spinner. Exists so a candidate always trains long enough for early-epoch noise to settle before the AUC comparison decides `not_improved`; a 1-epoch candidate would otherwise be rejected on fluke metrics. |
| `maximum_epochs` | `20` | Upper bound of the epoch spinner. |
| `default_validation_fraction` | `0.2` | Default hold-out fraction offered for validation cases. |
| `minimum_validation_cases_for_auto_selection` | `2` | A candidate is only auto-selected as the recommended model when at least this many validation cases were evaluated; with fewer, the result registers as `unverified`. |
| `minimum_mean_auc_improvement` | `0.0` | Candidate must beat the current model's mean trajectory AUC by at least this margin to qualify. |
| `maximum_severe_case_regression` | `0.2` | Quality gate: any validation case whose AUC drops more than this against the current model marks the candidate `not_improved` even when the mean improves. |
| `training_steps_per_epoch` | `50` | Trainer steps per epoch. |
| `validation_batches` | `8` | Validation batches per evaluation pass. |
| `gpu_lock_timeout_seconds` | `86400` | How long a queued training job waits for the shared GPU before failing. |
| `label_export_timeout_seconds` | `3600` | Deadline for the background-Mimics label export stage. |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup. |
| `keep_failed_training_artifacts` | `false` | Keep job artifacts after a failed training run for diagnosis (default: cleaned up). |
| `status_poll_seconds` | `1.0` | Model Center job-status polling interval. |

When a job fails, the Model Center progress page shows an aggregated
diagnosis (recorded error, failing stage, log tails, and cleanup summary)
with an **Open Job Folder** button. The same report is available from a
terminal:

```text
python tools/nninteractive_finetune_pipeline.py diagnose --job-dir <job folder>
```

## `flexict_config.json`

Controls the FlexiCT few-shot training pipeline (ViT backbone fine-tuning on
a handful of annotated cases, model registry, and job management). The
training recipe itself (optimizer, learning rates, batch size, fp32, TTA) is
locked to the validated few-shot configuration and is not configurable here.

| Key | Default | Purpose |
| --- | --- | --- |
| `workspace_dir` | `"flexict_models"` | FlexiCT workspace root (jobs, runtime nnU-Net folders, model registry). Relative paths resolve against the project root. |
| `flexict_dir` | `"integrations/flexict-finetune"` | The standalone FlexiCT integration repo (backbone, trainers, uncertainty tooling). |
| `pretrained_weights_dir` | `""` | Optional absolute override for the pretrained FlexiCT weights (needs `flexict_2d/model.safetensors` and `flexict_3d/model.safetensors`). Empty = use `integrations/flexict-finetune/weights/`. |
| `default_configuration` | `"auto"` | `2d`, `3d_fullres`, `pair` (2D+3D for active learning), or `auto` (<16GB GPU → 2D, otherwise pair). |
| `default_epochs` | `150` | Validated few-shot epoch count. |
| `default_mirror_disable_axes` | `""` | Mirror-augmentation axes to disable; `"1"` for single-sided organs (one kidney). |
| `default_val_cases` | `3` | Default held-out validation cases (best-checkpoint selection). |
| `dataset_id_first` | `750` | First id of the FlexiCT dataset band (750-799, kept apart from nnU-Net's 701+). |
| `default_uncertainty_method` | `"disagreement"` | Active-learning uncertainty method (2D+3D prediction disagreement). |
| `gpu_lock_timeout_seconds` | `86400` | How long a queued job waits for the shared GPU before failing. |
| `label_export_timeout_seconds` | `7200` | Deadline for the background-Mimics label export stage. |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup (0 disables sweeping). |
| `runtime_retention_days` | `30` | Retention for rebuildable `runtime/` Dataset folders (nnUNet_raw/nnUNet_preprocessed/nnUNet_results, 0 disables). Datasets referenced by registered models or non-terminal jobs are never removed; everything else is deleted once older than this. |
| `source_grid_cache_retention_days` | `30` | Retention for the rebuildable `cache/source_grid/` per-case training inputs (0 disables). Task trees of non-terminal jobs are never removed. |
| `status_poll_seconds` | `1.0` | Job-status polling interval. |

## `nnunet_config.json`

Controls the generic nnU-Net training/inference pipeline's job housekeeping.
The training recipe itself is managed by nnU-Net and is not configurable here.

| Key | Default | Purpose |
| --- | --- | --- |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup (0 disables sweeping). Terminal jobs (completed/failed/cancelled/abandoned) older than this are pruned to their `status.json` when any nnU-Net job finishes; registered models are never touched. |
| `runtime_retention_days` | `30` | Retention for rebuildable `runtime/` Dataset folders (nnUNet_raw/nnUNet_preprocessed/nnUNet_results, 0 disables). Datasets referenced by registered models or non-terminal jobs are never removed; everything else is deleted once older than this. |
| `source_grid_cache_retention_days` | `30` | Retention for the rebuildable `cache/source_grid/` per-case training inputs (0 disables). Task trees of non-terminal jobs are never removed. |

## Environment Variables

### Executables and Paths

| Variable | Purpose |
| --- | --- |
| `MIMICS_BACKGROUND_EXE` | Explicit executable for background import/export automation. Preferred troubleshooting override. |
| `MIMICS_EXE` | Legacy executable override, still accepted. |
| `MIMICS_BRIDGE_PYTHON` | Explicit bridge Python. Normally the packaged environment is used. |
| `MIMICS_BRIDGE_SCRIPT` | Explicit `mimics_bridge.py` path. |
| `MIMICS_IMPORT_RUNTIME_DIR` | Local import queue/control directory override. Do not place it on an unreliable network share. |
| `MIMICS_RESOURCE_LOCK_DIR` | Optional local directory for the shared GPU and background Mimics locks. Use the same value for Mimics and all external tools. UNC project roots otherwise fall back to a per-project directory under `%LOCALAPPDATA%` or `%TEMP%`. |
| `MIMICS_QT_PYTHONPATH` | Optional Qt module path for legacy internal UI fallback. |
| `MIMICS_REMOTE_CONFIG_DIR` | Override for the remote-compute config root (`servers.json`, `known_hosts`). Default: `%LOCALAPPDATA%\MimicsScript\remote_compute`. |

### Training worker seams (set by the pipelines, not by users)

These are written into worker environments by `flexict_pipeline.py` /
`nnunet_pipeline.py` / `nnunet_stage_worker.py`. They document the contract
between the Mimics layer and the training frameworks; users never set them.

| Variable | Purpose |
| --- | --- |
| `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` | Per-workspace nnU-Net folder triple. |
| `nnUNet_extTrainer` | Directory holding the external trainer classes (FlexiCT / nnU-Net extension), avoiding site-packages copies. |
| `nnUNet_compile` | Set to `0` (torch.compile off — reproducibility and startup time). |
| `FLEXICT_EXT_DIR` | `flexict` package dir for the FlexiCT trainer. |
| `FLEXICT2D_CKPT` / `FLEXICT3D_CKPT` | Pretrained FlexiCT backbone weights (safetensors). |
| `NUM_EPOCHS` | Overridden epochs for a FlexiCT run (smoke tests use 2). |
| `MIRROR_DISABLE_AXES` | Mirror augmentation axes to disable (e.g. `1` for a single-sided organ). |
| `MIMICS_REMOTE_GPU_LOCK` | GPU lock coordination for remote jobs. |

### Lifecycle and Diagnostics

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIMICS_AUTO_CLEANUP_ON_START` | `1` | Remove stale lock records whose owning process is gone. Does not kill healthy live tasks. |
| `MIMICS_USE_EVENT_TIMER` | unset | Opt in to Mimics event subscriptions instead of Win32 SetTimer for background monitors. Certain Mimics versions log Subscription.__del__ errors with the event path. |
| `MIMICS_IMPORT_AUTO_OPEN_MCS` | unset | Automatically open a completed imported project when supported. |
| `MIMICS_IMPORT_USE_MIMICS_LOG` | unset | Mirror verbose import diagnostics into the Mimics log panel. |
| `MIMICS_IMPORT_VERBOSE_LOG` | unset | Enable detailed import diagnostics. |
| `NNINTERACTIVE_MINIMUM_FREE_GPU_MEMORY_GB` | `4` | Override for the `minimum_free_gpu_memory_gb` config key (env wins). |

### Mask-at-Cursor Diagnostics

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIMICS_MASK_IDENTIFIER_USE_BBOX` | `1` | Use bounding boxes to skip masks that cannot contain the selected point. |
| `MIMICS_MASK_IDENTIFIER_VISIBLE_ONLY` | unset | Restrict checks to visible masks. Normally leave disabled. |
| `MIMICS_MASK_IDENTIFIER_ALLOW_PARTIAL` | unset | Permit partial scans when a configured limit is reached. |
| `MIMICS_MASK_IDENTIFIER_MAX_MASKS_PER_CLICK` | `0` | Optional per-click mask limit; zero means unlimited. |
| `MIMICS_MASK_IDENTIFIER_MAX_SECONDS_PER_CLICK` | `0` | Optional per-click time limit; zero means unlimited. |
| `MIMICS_MASK_IDENTIFIER_MAX_CACHED_BUFFERS` | `4` | Maximum cached mask voxel buffers. |

## Background Mimics Limitation

Batch operations on multiple saved `.mcs` files require a separate background
Mimics executable and an available license because the foreground annotation
project must not be replaced or blocked. The software does not silently fall
back to opening projects synchronously in the annotation window.

When background Mimics is unavailable:

1. Configure `MIMICS_BACKGROUND_EXE`, or
2. Export labels separately and turn off **Refresh labels from saved .mcs before training**.

Current-project mask export remains available through **Export Masks** and is
split into timer-driven steps so GUI updates can occur between mask-buffer
reads.

## Portable AI Models

Trained model runtime files use paths relative to their own model manifest or
model workspace. Copying a complete workspace therefore does not preserve a
dependency on the source computer's drive letter or user directory.

The nnInteractive **Model Versions** tab can export the current model to a
self-contained `.zip` and import that package on another workstation. Import
validates the package before publishing files and rebuilds the target
machine's local registry.

The same operations are available without Mimics:

```text
python tools/ai_model_bundle.py export-nninteractive --workspace <dir> --task-id <task> --output <file.zip>
python tools/ai_model_bundle.py import-nninteractive --workspace <dir> --bundle <file.zip> --set-current
```

## ScribblePrompt

ScribblePrompt uses the official UNet checkpoint at
`integrations/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt`.
The default device is `cpu`, avoiding competition with training and
nnInteractive for GPU memory. Set `scribbleprompt.device` to `auto` or `cuda`
only when shared GPU-lock waiting is acceptable.

| Variable | Purpose |
| --- | --- |
| `MIMICS_INTERACTIVE_ALGORITHMS_CONFIG` | Override the configuration JSON path. |
| `SCRIBBLEPROMPT_CHECKPOINT` | Override the official ScribblePrompt UNet checkpoint path. |

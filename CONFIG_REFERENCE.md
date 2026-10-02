# Mimics-Script Configuration Reference

This document lists the supported user-facing configuration files and
environment variables. Paths may be absolute or relative to the project root
unless noted otherwise.

## How configuration is layered

Each config JSON carries **only the keys an annotator or an administrator
plausibly changes**. Everything else — timeouts, poll intervals, cache
retention, internal mechanism switches — is built into the code as defaults.
Removing a key from a JSON file never breaks anything: the code falls back to
the same default, and a key re-added later overrides it again. Keys fall into
three groups:

1. **Annotator keys** — shown in Mimics under **Admin > Edit Configs** with a
   one-line explanation each.
2. **Ops keys** — deployment-specific (workspace/model folders); edited by
   hand only when moving an installation.
3. Internal keys — not in the JSON files at all; the tables below document
   them for operators who need an escape hatch.

## `mimics_io_config.json`

| Key | Default | Purpose |
| --- | --- | --- |
| `mimics_output_dir` | `""` | Default folder for generated `.mcs` projects. An empty value uses `<dataset>/mcs_output`. This setting does not redirect mask exports, training runs, or logs. |
| `mimics_background_exe` | auto-detect | Path to `materialise.exe` used for background batch operations. Empty = auto-detect from the running Mimics. |
| `mimics_buffer_axes` | `[0, 1, 2]` | Optional advanced mapping from external array axes to the Mimics voxel buffer. Change only after real-data orientation validation. |
| `mimics_buffer_flips` | `[false, false, false]` | Optional advanced flips paired with `mimics_buffer_axes`. Change only after real-data orientation validation. |

## `interactive_algorithms_config.json`

Controls the interactive segmentation algorithms run outside Mimics
(ScribblePrompt).

| Key | Default | Purpose |
| --- | --- | --- |
| `scribbleprompt.timeout_seconds` | `900` | Give up waiting for a ScribblePrompt prediction after this many seconds. |
| `scribbleprompt.device` | `"cpu"` | `cpu` (default; never competes with training for GPU memory) or `cuda`. |

Internal keys (code defaults; add to the file only as an escape hatch):
`poll_seconds` (0.25), `job_retention_days` (3), `job_max_terminal` (20),
`scribbleprompt.input_size` (128), `scribbleprompt.prior_logit_magnitude`
(6.0), `scribbleprompt.gpu_lock_timeout_seconds` (120). The checkpoint path
comes from `SCRIBBLEPROMPT_CHECKPOINT` or the bundled
`integrations/ScribblePrompt/checkpoints/` location.

## `window_level_presets.json`

CT window/level presets matched against Mask names (Review menu). Editable
from Mimics via **Review > Window Edit Presets**; no hand editing required.

Each entry: `name`, `width`, `level`, `keywords` (matched case-insensitively
against the selected Mask name, first keyword hit wins), and `source`
(citation for the W/L values). Ships with 5 presets: Lung, Abdomen / Soft
Tissue, Bone, Skull / Cranium, Vessel / Heart. State (undo/last applied)
lives in `.mimics_runtime/window_level_state.json`; user edits are saved back
to this file.

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
| `gpu_busy_policy` | What happens at launch when the selected GPU already looks busy on this shared server: `block` (default, refuse to start training), `warn` (start and record a note), `off` (skip the check). |
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
  discovery-equivalence test in `tests/test_all.py`.
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
(`99_Admin/04_Undo_Last_Import.py`):

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

`01_Data/01_Import_Data.py` opens an always-on-top external window
(`tools/import_drop_window.py`) that accepts dragged files, case folders,
dataset folders, pasted paths, or hand-picked files/folders, classifies them
against the dataset
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
| `workspace_dir` | `"nninteractive_task_models"` | Task-model workspace root (custom trained models). Ops key: change only when relocating the workspace. |
| `device` | `"auto"` | Inference device: `auto` = first free GPU, `cpu` = force CPU (use when the GPU is needed by a training job). |
| `auto_start_server` | `true` | Start the nnInteractive server automatically when an annotation session begins. |
| `existing_mask_result_mode` | `"ask"` | Ask whether to update the selected mask or create an editable copy (`ask` / `replace` / `copy`). |
| `keep_server_warm_after_session` | `true` | Keep the server alive between annotation sessions so the next session starts faster. |
| `server_idle_timeout_seconds` | `300` | The warm server shuts down after this many idle seconds. |
| `minimum_free_gpu_memory_gb` | `4` | Annotation sessions refuse to start when the GPU has less free memory than this, with a "close other GPU programs" message instead of a mid-start OOM. |

Internal keys (code defaults; add to the file only as an escape hatch):
`model_dir` (unset), `image_input_mode` ("mimics"),
`task_model_image_input_mode` ("auto"),
`prefer_source_image_for_nninteractive` (false),
`fallback_to_mimics_buffer_when_source_unavailable` (true),
`fallback_to_source_when_mimics_export_fails` (false),
`async_start_worker_before_prompt` (true), `async_reuse_image_worker` (true),
`incremental_interaction_replay` (true),
`async_worker_idle_timeout_seconds` (900),
`server_startup_timeout_seconds` (900), `set_image_timeout_seconds` (1800),
`prediction_timeout_seconds` (1800), `bridge_timeout_seconds` (4620),
`gpu_lock_timeout_seconds` (30), `async_poll_seconds` (0.25),
`async_result_poll_seconds` (0.25), `async_job_retention_days` (3),
`async_job_max_terminal` (20), `source_cache_retention_days` (7),
`source_cache_max_entries` (12).

## `nninteractive_finetune_config.json`

Controls the nnInteractive task fine-tuning pipeline (Model Center, label
export, training jobs, and quality gating).

| Key | Default | Purpose |
| --- | --- | --- |
| `workspace_dir` | `"nninteractive_task_models"` | Task-model workspace root (tasks, models, registry). Ops key. |
| `default_epochs` | `10` | Initial epoch spinner value in the training dialog (range 4–20). |
| `default_validation_fraction` | `0.2` | Default hold-out fraction offered for validation cases. |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup. |
| `keep_failed_training_artifacts` | `false` | Keep job artifacts after a failed training run for diagnosis (default: cleaned up). |

Internal keys (code defaults; add to the file only as an escape hatch):
`official_model_dir` (auto-detected under `python_env`/`nninteractive_env`),
`minimum_epochs` (4), `maximum_epochs` (20),
`minimum_validation_cases_for_auto_selection` (2),
`minimum_mean_auc_improvement` (0.0), `maximum_severe_case_regression` (0.2),
`training_steps_per_epoch` (50), `validation_batches` (8),
`gpu_lock_timeout_seconds` (86400), `label_export_timeout_seconds` (3600),
`prepared_cache_retention_days` (30), `status_poll_seconds` (1.0).

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
| `workspace_dir` | `"flexict_models"` | FlexiCT workspace root (jobs, runtime nnU-Net folders, model registry). Ops key: change only when relocating the workspace. |
| `flexict_dir` | `"integrations/flexict-finetune"` | The standalone FlexiCT integration repo. Ops key. |
| `pretrained_weights_dir` | `""` | Optional absolute override for the pretrained FlexiCT weights (needs `flexict_2d/model.safetensors` and `flexict_3d/model.safetensors`). Empty = use `integrations/flexict-finetune/weights/`. |
| `default_configuration` | `"auto"` | `2d`, `3d_fullres`, `pair` (2D+3D for active learning), or `auto` (<16GB GPU → 2D, otherwise pair). |
| `default_epochs` | `150` | Validated few-shot epoch count. |
| `default_val_cases` | `3` | Default held-out validation cases (best-checkpoint selection). |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup (0 disables sweeping). |

Internal keys (code defaults; add to the file only as an escape hatch):
`default_mirror_disable_axes` (""), `dataset_id_first` (750),
`default_uncertainty_method` ("disagreement"), `gpu_lock_timeout_seconds`
(86400), `label_export_timeout_seconds` (7200), `status_poll_seconds` (1.0).

## `nnunet_config.json`

Controls the generic nnU-Net training/inference pipeline's job housekeeping.
The training recipe itself is managed by nnU-Net and is not configurable here.
All three keys are internal (code defaults; add to the file only as an escape
hatch — none are annotator-facing).

| Key | Default | Purpose |
| --- | --- | --- |
| `job_retention_days` | `30` | Terminal job folder retention before cleanup (0 disables sweeping). Terminal jobs (completed/failed/cancelled/abandoned) older than this are pruned to their `status.json` when any nnU-Net job finishes; registered models are never touched. |
| `runtime_retention_days` | `30` | Retention for rebuildable `runtime/` Dataset folders (nnUNet_raw/nnUNet_preprocessed/nnUNet_results, 0 disables). Datasets referenced by registered models or non-terminal jobs are never removed; everything else is deleted once older than this. |
| `source_grid_cache_retention_days` | `30` | Retention for the rebuildable `cache/source_grid/` per-case training inputs (0 disables). Task trees of non-terminal jobs are never removed. |

### Train/validation split contract

The nnU-Net and FlexiCT training pipelines split cases into five rotating
folds keyed by `split_seed` and `validation_fraction` (both editable in the
training UI). Two standing guarantees:

- **Patient grouping** — a training request may carry an optional
  `patient_groups` mapping (`case_id` → patient/subject id). When present,
  cases of one patient always land on the same side of the split, so a
  patient's multi-phase or follow-up scans never appear in both train and
  validation. Without the mapping the split assumes each case is an
  independent patient.
- **Frozen validation set** — rebuilding a dataset that already published a
  fold-0 validation split keeps those cases in validation on every fold.
  Adding cases to a dataset never moves previously-validated cases into
  training, so quality numbers stay comparable across rebuilds.

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

Current-project mask export remains available through **Export Masks**
and is split into timer-driven steps so GUI updates can occur between
mask-buffer reads.

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

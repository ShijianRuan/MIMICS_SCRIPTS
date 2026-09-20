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

## `fewshot_config.json`

### Common Training Defaults

| Key | Default | Purpose |
| --- | --- | --- |
| `default_training_profile` | `"balanced"` | Initial profile loaded by the training window. |
| `default_epochs` | `20` | Default training epochs. |
| `default_lr` | `0.001` | Default learning rate. |
| `default_lr_scheduler` | `"cosine"` | Initial learning-rate schedule. |
| `default_warmup_epochs` | `0` | Warmup epochs where supported; zero disables warmup. |
| `default_weight_decay` | `0.0001` | Default optimizer weight decay. |
| `default_grad_accumulation` | `1` | Effective-batch accumulation. Physical batch size remains one for variable-depth volumes unless a compatible 2D policy is selected. |
| `default_img_size` | `"256,256"` | In-plane training size. |
| `default_decoder` | `"auto"` | Let the selected strategy and dimensionality choose a compatible decoder. |
| `default_finetune_method` | `"frozen"` | Initial backbone fine-tuning method. |
| `default_model_scale` | `"vits16"` | Initial DINOv3 backbone scale. |
| `default_modality` | `"auto"` | Infer CT/MR preprocessing from the selected training data when possible. |
| `default_val_fraction` | `0.2` | Default validation fraction. |
| `default_min_samples` | `1` | Minimum usable training samples. |
| `default_min_val_samples` | `1` | Minimum validation samples when validation is enabled. |
| `default_max_samples` | `0` | Maximum selected samples; zero means no limit. |
| `default_export_labels_before_training` | `true` | Refresh labels from saved `.mcs` projects before training. |
| `default_keep_last_checkpoints` | `2` | Number of recent checkpoints retained in addition to the best model. |
| `default_keep_materialized_dataset` | `false` | Keep the temporary `imagesTr`/`labelsTr` dataset after the job. |

Generated DINOv3 experiment YAML uses model-space `ZYX` axes for
`inference.tta_axes` and `augmentation.flip_axes`. Axis `2` is the canonical
RAS left/right axis. Standard Mimics profiles do not mirror it. A deliberately
laterality-invariant custom experiment must set
`augmentation.allow_left_right_flip: true` before using axis `2`; otherwise
training fails closed. Native cached-slice models derive their safe in-plane
flip axis from each source affine and store it in the feature-cache manifest.

### Saved Mask Name Mapping

`organ_mask_aliases` maps one training target to accepted names in saved
`.mcs` projects:

```json
{
  "organ_mask_aliases": {
    "liver": ["liver_seg", "Segment_Liver", "肝"],
    "adrenal_gland_right": ["right_adrenal", "adrenal_right"]
  }
}
```

The training window shows the resolved names in **Saved mask names** and lets
the user edit them for that run. Matching is case-insensitive and normalizes
spaces, hyphens, slashes, and periods to underscores. It is intentionally not
fuzzy: automatic substring matching could silently select the wrong organ.

Before any voxel buffer is read, background Mimics opens every selected
project and validates the names. Each project must contain exactly one matching
mask. Missing or ambiguous cases fail the whole fresh-label export before
training starts. The exported NIfTI label is renamed to the training target,
so an accepted `liver_seg` mask becomes `liver.nii.gz`.

### Runtime and Retention

| Key | Default | Purpose |
| --- | --- | --- |
| `gpu_lock_timeout_seconds` | `900` | Maximum wait for the shared GPU before training fails. |
| `inference_gpu_lock_timeout_seconds` | `3600` | Maximum wait for the shared GPU before inference fails. |
| `background_mimics_lock_timeout_seconds` | `1800` | Maximum wait for the shared background Mimics license/process. |
| `terminal_job_retention_days` | `30` | Retention for terminal job records. |
| `max_terminal_job_records` | `100` | Maximum terminal job records retained. |
| `failed_run_retention_days` | `7` | Retention for failed disposable run directories. |
| `keep_failed_training_artifacts` | `false` | Keep large rebuildable datasets, caches, checkpoints, and partial models after a failed DINOv3 or nnInteractive fine-tuning job. Logs, status, metrics, and configuration are always retained. |
| `completed_run_log_retention_days` | `30` | Retention for completed text logs. |
| `setup_context_retention_days` | `7` | Retention for abandoned setup contexts. |
| `keep_training_experiment_artifacts` | `false` | Keep disposable DINOv3 experiment folders after registration. |

### External UI

| Key | Default | Purpose |
| --- | --- | --- |
| `advanced_ui_mode` | `"external"` | Use the external PySide6 training window. |
| `advanced_ui_fallback_to_internal` | `true` | Permit the limited internal fallback if the external window cannot start. |
| `status_ui_mode` | `"external"` | Use the external PySide6 status viewer. |
| `status_ui_fallback_to_text` | `true` | Show concise Mimics text status if the external viewer cannot start. |
| `dinov3_project` | `"external/dinov3-medical-seg"` | DINOv3 project path. |
| `python` | `""` | Optional explicit external Python. The packaged `nninteractive_env` is preferred. |

`training_profiles` contains reusable initial values only. Values remain
editable in the training window.

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

## Environment Variables

### Executables and Paths

| Variable | Purpose |
| --- | --- |
| `MIMICS_BACKGROUND_EXE` | Explicit executable for background import/export automation. Preferred troubleshooting override. |
| `MIMICS_EXE` | Legacy executable override, still accepted. |
| `MIMICS_BRIDGE_PYTHON` | Explicit bridge Python. Normally the packaged environment is used. |
| `MIMICS_BRIDGE_SCRIPT` | Explicit `mimics_bridge.py` path. |
| `MIMICS_FEWSHOT_PYTHON` | Explicit DINOv3 training/inference Python. |
| `MIMICS_FEWSHOT_DINOV3_ROOT` | Explicit DINOv3 project root. |
| `MIMICS_IMPORT_RUNTIME_DIR` | Local import queue/control directory override. Do not place it on an unreliable network share. |
| `MIMICS_RESOURCE_LOCK_DIR` | Optional local directory for the shared GPU and background Mimics locks. Use the same value for Mimics and all external tools. UNC project roots otherwise fall back to a per-project directory under `%LOCALAPPDATA%` or `%TEMP%`. |
| `MIMICS_QT_PYTHONPATH` | Optional Qt module path for legacy internal UI fallback. |
| `MIMICS_DINOV3_GUI_BACKEND` | Force `pyside6`, `tkinter`, or `auto` for external DINOv3 windows. |

### Lifecycle and Diagnostics

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIMICS_AUTO_CLEANUP_ON_START` | `1` | Remove stale lock records whose owning process is gone. Does not kill healthy live tasks. |
| `MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START` | unset | Opt-in termination of owned stale services at startup. Use only for recovery. |
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

The nnInteractive **Model Versions** tab and DINOv3 **Show Status** window can
export the current model to a self-contained `.zip` and import that package on
another workstation. Import validates the package before publishing files and
rebuilds the target machine's local registry.

The same operations are available without Mimics:

```text
python tools/ai_model_bundle.py export-nninteractive --workspace <dir> --task-id <task> --output <file.zip>
python tools/ai_model_bundle.py import-nninteractive --workspace <dir> --bundle <file.zip> --set-current
python tools/ai_model_bundle.py export-dinov3 --manifest <manifest.json> --output <file.zip>
python tools/ai_model_bundle.py import-dinov3 --workspace <dir> --bundle <file.zip> --set-latest
```

DINOv3 packages contain the trained decoder or adaptation checkpoint and its
flattened inference configuration. Standard pretrained encoders remain shared
runtime assets under `external/dinov3-medical-seg/models` and must exist on the
target installation, as verified by environment setup and package checks.

## ScribblePrompt

ScribblePrompt uses the official UNet checkpoint at
`external/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt`.
The default device is `cpu`, avoiding competition with training and
nnInteractive for GPU memory. Set `scribbleprompt.device` to `auto` or `cuda`
only when shared GPU-lock waiting is acceptable.

| Variable | Purpose |
| --- | --- |
| `MIMICS_INTERACTIVE_ALGORITHMS_CONFIG` | Override the configuration JSON path. |
| `SCRIBBLEPROMPT_CHECKPOINT` | Override the official ScribblePrompt UNet checkpoint path. |

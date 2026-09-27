# nnU-Net integration for Mimics

## Product workflow

The Scripting Library exposes three entries under `02_AI/nnUNet`:

1. **01 Train Model** opens one external PySide6 window.
2. **02 Predict Current Case** opens a compatible-model selector.
3. **03 Show Status and Models** shows the selected task's runs, live log,
   curve, and registered models, and requests a bounded local or remote
   shutdown through its Stop control.

The external windows are separate processes. Mimics performs no dataset scan,
network connection, preprocessing, training, or inference on its GUI thread.
It only writes a small request, launches the external process, polls a status
JSON with a timer, and applies a verified result one Mask at a time.

The managed layer reuses the proven planning, training, and prediction calls in
`integrations/nnunet_segmentation_workflow`. It adds the contracts that Mimics
needs: source-grid label preparation, background lifecycle management, portable
model registration, remote execution, and guarded result application.

## Task and model hierarchy

The managed object is a **segmentation task/model**, not an organ. One nnU-Net
model can emit many output labels, such as an abdomen model containing liver,
spleen, kidneys, adrenals, and other structures. The Mimics training window
therefore keeps the model task name separate from its editable output-label
table.

Each output-label row has one integer class ID, one or more accepted source Mask
names, and an explicit source rule. **Match one** treats the names as alternate
names for the same structure and rejects ambiguous multiple matches.
**Union all** aligns every matching source Mask to the original-image grid and merges
them into one output class, matching the grouped coarse-localization semantics
in the existing workflow. Repeated class IDs in a flat map are imported with
the same union semantics. Existing flat and grouped tables from
`ModelMap.toml`, as well as labels from an nnU-Net `dataset.json`, can be loaded
with **Import Label Set**.

Standard nnU-Net does not provide partial-label supervision. The training UI
therefore makes incomplete-case handling explicit.
**Require every configured label** is the safe default and skips a case with any
missing class. **Treat a missing Mask as background** matches the legacy converter, but should only be
used when an absent file truly means that class is absent, not merely
unannotated.

Selecting a Mask before prediction is only a ranking hint: models containing
that label appear first, but no other multi-label models are hidden. The chosen
model predicts and applies all labels declared by its manifest. Status is
organized by the model task, not by the active Mask.

The standalone workflow also contains `multimodel_predict_and_merge`, which is
a separate composition layer for combining several independently trained
multi-label models. A managed Mimics prediction currently executes one
registered multi-label model at a time; this must not be confused with
single-organ inference.

## Training data contract

One task may contain one or more mutually exclusive labels. Each label has:

- a stable output name;
- an integer ID from 1 to 255;
- one or more accepted source Mask names.

Three label sources are supported:

- **Masks in each dataset case** reads the original case `segmentations`;
- **Previously exported Masks** consumes a user-selected export folder;
- **Refresh from saved .mcs projects** exports only the configured names from
  saved projects in a background Mimics process.

Every source image is materialized as NIfTI without resampling. Every label is
then mapped to that exact shape and voxel-to-RAS affine. Equal grids are copied,
axis permutations/flips use exact reindexing, and genuinely different physical
grids use label-safe nearest-neighbor resampling. A non-empty label that becomes
empty is rejected. Overlapping classes fail by default because one nnU-Net
multiclass voxel cannot represent two Mimics Masks.

The managed converter does not use the legacy framework behavior that chooses
the first arbitrary file in a case directory.

## Exposed parameters

The window groups settings by meaning rather than “basic” and “advanced”:

- **Data:** task, dataset ID, modality, image root, label source, label mapping,
  overlap policy, and model library.
- **Planning:** 2D, 3D full resolution, or 3D low resolution; automatic or
  explicit spacing, patch size, and batch size.
- **Training:** trainer, epochs, fold, validation fraction, preprocessing
  workers, GPU count, optional local GPU IDs, pretrained checkpoint,
  continuation, and inference TTA.
- **Compute:** this workstation or a saved SSH/Docker server and GPU.

The interface does not expose unsupported combinations. Arbitrary epochs are
enabled for `MimicsNNUNetTrainer` and
`MimicsNNUNetTrainerNoMirroring`. The standalone TOML workflow maps
`nnUNetTrainer` and `nnUNetTrainerNoMirroring` to those behavior-equivalent
configurable classes, so `TRAIN.epoch` now changes `num_epochs`. An unknown
custom Trainer with an explicit epoch setting is rejected instead of silently
ignoring the value. Official or arbitrary custom Trainers selected directly in
the Mimics UI continue to own their training length, so the epoch control is
disabled for them.

## Workflow notes

The legacy standalone TOML CLI (`AutoSegmentationFramework.py`, `Config_*.toml`)
was removed together with its dead convert/evaluate stages; the managed Mimics
window is the only entry point. The surviving stage backends are imported
directly by `tools/nnunet_stage_worker.py`.

Manual spacing is disabled for the 2D configuration because nnU-Net's planner
does not apply `overwrite_target_spacing` to an independent 2D plan. Patch
dimensions are shown as nnU-Net array axes rather than patient X/Y/Z axes.
Continuing an interrupted fold disables pretrained-weight selection because
the two nnU-Net startup modes are mutually exclusive.

`3d_cascade_fullres` is intentionally not exposed as a single selection. It
requires a completed low-resolution fold and a coordinated cascade workflow;
silently treating it as an independent model would create invalid jobs.

Learning rate, optimizer, augmentation, network topology, and deep-supervision
policy remain owned by the selected Trainer. Exposing independent controls for
them would allow combinations that an arbitrary Trainer may ignore or reject.
The Trainer field is editable, so a validated custom nnU-Net Trainer can still
be selected without adding misleading generic switches.

The window suggests the first unused Dataset ID from 701 onward. Dataset IDs
are unique within one model library because nnU-Net resolves data by number
across raw, preprocessed, and result roots. The backend rejects a conflicting
ID before preprocessing and serializes preparation for the same ID with a
cancellable dataset lock. Different Dataset IDs can still prepare concurrently.

## Local execution

Heavy stages execute through `tools/nnunet_stage_worker.py` in the bundled
`nninteractive_env` Python:

1. discover and validate cases;
2. create deterministic `nnUNet_raw` data and five reproducible folds;
3. fingerprint, plan, and preprocess, or reuse a matching cache;
4. train the requested fold;
5. verify `checkpoint_final.pth`;
6. copy a portable model bundle and register it locally and globally.

Each stage has a child PID, combined log, status, control file, and process-tree
termination path. A stop request becomes `Cancelled` only after the child is
reaped. Failures preserve the job folder and diagnostic log.

The shared GPU lock is transferred from the controller to the actual training
or inference worker immediately after it starts. The status records both the
worker PID and its OS process-start marker. If the controller is killed, the
worker therefore continues to own the lock instead of being mistaken for a
stale process. **Stop** normally asks the live controller to shut down cleanly;
if the controller is gone, it terminates the verified orphan worker directly.
A reused PID without the recorded start marker is never killed.
The worker also waits behind a one-use start gate until lock transfer and status
persistence have both succeeded. A controller failure during the short spawn-to-
transfer interval therefore leaves a non-GPU child that times out and exits,
not an untracked worker that can race a new task.

Completed logs are bounded to 32 MB. Compaction retains the startup section and
the most recent diagnostics; training curves and terminal status remain separate
files. This prevents a long-running workstation from accumulating unbounded
per-job console output.

Preprocessing is reused only when the source-grid dataset and planning options
have the same fingerprint. On a mismatch, the old generated dataset directory
is removed before preprocessing so deleted cases cannot survive as stale
arrays. Training and inference share the repository-wide local GPU lock with
nnInteractive. Waiting is cancellable and visible in status; an idle
nnInteractive server is asked to release the GPU. Remote containers use the
server-side GPU scheduler instead and do not acquire this local lock again.

## Remote execution

Remote mode reuses the existing SSH profile and unified Docker image. The
client prepares source-grid cases locally, creates deterministic per-case
archives, uploads only missing content hashes, and starts one disposable
container on the selected GPU.

The container receives no SSH service and exposes no port. It mounts:

- the isolated job at `/job`;
- read-only base models at `/models`;
- GPU locks at `/remote-locks`;
- per-user reusable data at `/remote-cache`.

nnU-Net raw and preprocessed roots include the complete archived dataset
fingerprint. Unchanged datasets reuse preprocessing; any changed image or label
uses a new namespace. Successful models are checksummed, downloaded, registered
locally, and remain usable without the server. Containers and job staging are
removed after confirmed completion or cancellation; caches and the image remain.

Remote inference uses the same controller. The original image and selected
model are transferred through content-addressed archives, so unchanged assets
do not need to be uploaded again.

Remote nnU-Net raw and preprocessed namespaces are isolated by complete
dataset fingerprint for training and by Dataset ID plus source fingerprint for
inference. Reusable namespaces are retained for 30 days by default. Cleanup is
deferred while any container owned by the same server profile is active, and
paths outside that user's prepared-cache root are never removed. When data
caching is disabled, the job-specific namespace is removed after a confirmed
terminal state.

The controller synchronizes the nnU-Net pipeline log rather than only the
container wrapper log, so epoch output is visible in the local Status window.
For training it also downloads an updated `progress.png` at a bounded interval.
If the local SSH controller dies after a remote job may have launched, the task
becomes `Orphaned Remote` instead of falsely becoming failed; **Stop** reconnects
and removes only that owned container. A controller that dies before remote
launch is terminally failed because no remote GPU resource can exist yet.

## Inference and Mimics application

The selected model's own `plans.json`, `dataset.json`, trainer, configuration,
and available folds control inference. The interface does not ask the user to
re-enter training preprocessing parameters.

Input files, MHD/MHA/NRRD, and DICOM folders are materialized without changing
their physical grid. The output must match the materialized source image shape
and affine before the job can complete.

Before inference begins, the materialized source image must also match the
source shape and voxel-to-RAS affine recorded in the open Mimics project. A
relocated but identical source remains valid; a changed or incorrectly relinked
source fails before GPU work rather than producing a displaced Mask.

New model manifests also store a training-data profile: modality, channel and
spatial dimensionality, and the observed spacing and physical field-of-view
ranges. Task, CT/MR modality, channel count, and dimensionality mismatches fail
before GPU work. A strongly out-of-distribution spacing or field of view is a
visible warning rather than a hard failure. Exact training shape, origin, or
affine equality is deliberately not required: nnU-Net is expected to predict
new patients with different grids, and its own plans perform the model-space
resampling. Geometry alone cannot prove that a CT contains the intended organ,
so model choice remains explicit in the prediction window.

Mimics records its live launch-time voxel grid. The bridge splits a multiclass
prediction into binary labels and maps each label from source RAS space to that
verified Mimics grid. If another project or image is active, the result waits
instead of being applied to the wrong case.

When ready, the user chooses once:

- **Update Matching Masks** replaces only unchanged Masks whose names match a
  model label or alias; edited or missing Masks become copies.
- **Create Editable Copies** creates `AI_<label>` Masks and preserves all
  existing work.

Application is journaled per output label in the job status before and after a
Mask buffer is changed. The planned Mask name/GUID and each completed label are
therefore available after a Mimics restart. Recovery reuses the same planned
Mask and skips labels already committed, instead of creating a second AI copy
because the old in-memory application list was lost.

Pending completed predictions are rediscovered when any nnU-Net entry runs, so
closing and reopening a Mimics session does not discard a finished result.
The Stop entry can also cancel a pending conversion/application after inference
has completed. Conversion has a 600-second deadline and its owned bridge process
is terminated asynchronously; this does not terminate Mimics.

## Supported input shapes and formats

The managed single-channel workflow accepts case images discoverable by the
existing converter, including NIfTI, MHD/MHA, NRRD, and DICOM series folders.
Images are materialized to NIfTI without changing their physical grid. Labels
may use NIfTI, MHD/MHA, or NRRD and are mapped by their voxel-to-RAS affine.

This integration currently creates one nnU-Net input channel. Multi-label
segmentation is supported, but multimodal multi-channel input such as paired CT
and PET is not exposed until channel pairing and missing-channel validation are
implemented end to end.

## Lifecycle and interaction boundaries

- Dataset discovery, hashing, `.mcs` label export, preprocessing, training,
  SSH, Docker, inference, and result conversion run outside the Mimics GUI
  thread.
- Mimics timers only read small JSON files and apply one prepared Mask buffer per
  tick.
- The only intentional modal interaction is the final choice between updating
  matching Masks and creating editable copies, plus explicit Stop confirmation.
- Local training and inference share the repository GPU lock with
  nnInteractive. Remote jobs use the selected server GPU queue and do not take
  the local GPU lock.
- Local setup failure, local worker death, remote-controller loss, remote
  container failure, cancellation, and conversion timeout all have distinct
  visible states and retained diagnostics.

## Portability

Model bundles contain the nnU-Net model directory plus
`mimics_model_manifest.json`. Registries store absolute paths for the current
machine, while the manifest is also inside the model folder. Copying the full
model folder into `<model library>/models/<task>/<model>` on another machine is
sufficient: the model loader discovers the embedded manifest and resolves its
new location even if the copied registry still contains old absolute paths.
Inference does not depend on the original training dataset or `.mcs` path.

Projects used for training still need either reachable original images or a
portable exported-Mask dataset because source physical intensities are not
reconstructed from a display buffer.

## Environment requirement

The shared external Python environment must include `nnunetv2>=2.8.1,<2.9`.
`tools/setup_env.py`, portable-package verification, the offline setup check,
and the remote Docker preflight all verify this dependency. An offline Windows
bundle must therefore include nnU-Net and all of its dependency wheels before
running `setup_offline.bat`.

The remote image preflight verifies both `nnunetv2` and the bundled
`MimicsNNUNetTrainer`; an image that can import nnU-Net but cannot discover the
managed Trainer is rejected before data upload.

## Validation status

Automated tests cover request constraints, alias ambiguity, Dataset ID
collisions, deterministic five-fold splits, exact orientation changes,
multiclass overlap rejection, source-image and prediction affine checks, model
relocation, fold-all inference, local/remote orphan states, worker-owned GPU
locks, verified orphan shutdown, stop ordering, cache identity and corruption
checks, remote cache isolation/retention, download rollback, training-profile
compatibility, remote log/curve synchronization, and crash-safe Mask
application recovery.

The remaining acceptance checks require the target Windows workstation:

1. Run a short 2D and 3D fold with the installed CUDA build and confirm the
   bundled Trainer is discovered.
2. Train from dataset Masks, exported Masks, and refreshed `.mcs` Masks on at
   least one oblique and one axis-permuted case.
3. Predict locally and remotely, then verify every output label against the
   source NIfTI and its applied Mimics Mask.
4. Stop during `.mcs` export, preprocessing, GPU wait, training, remote upload,
   remote training, inference, and result conversion; confirm no owned child or
   container remains.
5. Copy a complete model folder to another model library and confirm it is
   discovered and predicts without the original training dataset.

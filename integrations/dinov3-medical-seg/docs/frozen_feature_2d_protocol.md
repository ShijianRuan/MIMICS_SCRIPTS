# Frozen Feature 2D Training Protocol

## Purpose

This document defines the default frozen-feature training method integrated
into Mimics-Script. The Mimics UI uses neutral names and does not expose the
name of the software used as a compatibility reference.

The implementation targets functional protocol equivalence:

- the same source image and source-grid label are used;
- preprocessing, encoder input, decoder topology, loss, augmentation,
  scheduling, validation TTA, and best-model selection follow the verified
  reference behavior;
- inference restores the prediction to the exact source NIfTI grid;
- training and inference use the same preprocessing and model contract.

PyTorch and tinygrad do not guarantee byte-identical trained weights, even
with the same nominal seed, because random-number generators, convolution
kernels, optimizer arithmetic, and shuffle implementations differ. The
supported acceptance criterion is numerical component parity plus equivalent
segmentation behavior. Byte-identical weights would require running the same
framework, kernels, and random stream as the reference implementation.

## Verified Protocol

The protocol was checked against readable source from the locally installed
macOS application and the recovered Windows material under
`external/RONDON_analysis`.

### Input and preprocessing

1. Read each 3D image and retain its native voxel index order.
2. Convert the array to `Z,Y,X` slice order without canonical reorientation.
3. Apply the model-specific casewise normalization declared by the installed
   training plan.
4. Resample only the in-plane `X/Y` dimensions to `256x256`; preserve the
   original slice count and through-plane spacing.
5. Resize every label slice with nearest-neighbor interpolation.
6. Repeat the normalized grayscale slice to three channels.
7. Send the three channels directly to the model-specific ONNX encoder.
8. Encode every slice once and cache the final `384x16x16` feature map.

The currently installed model-specific training path was instrumented directly.
It sends `N,3,256,256` to the encoder and receives `N,384,16,16`. The exact
downloaded ONNX artifact declares the same fixed input. ONNX Runtime accepts
`256x256` for that file and rejects both `224x224` and `512x512`.

The readable `_commands/_encoder.py` source genuinely resizes to `512x512`,
applies ImageNet normalization, and expects a `32x32` feature grid. Recovered
Windows material describes the same contract. That is a separate generic
encoder path/build contract, not the model-specific
`_tauri_app.models.DINOV3VITSP16` path used by the locally installed training
workflow. A matching 512-capable ONNX can use that protocol, but the verified
fixed-256 artifact cannot.
The verified model identity is also pinned by SHA-256:
`fb5c06a4487a3fa8d8bdcb3dcb68a55ffb23310d00d13809f52651442054db45`.
Training and inference reject a same-named but different ONNX file.

The same model directory may also contain `model.safetensors`. The PyTorch
backend validates that file against SHA-256
`4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`.
It accepts any positive multiple-of-16 spatial size supported by the ViT
architecture; local forward checks passed at 224, 256, and 512.

The bundled ONNX and HuggingFace safetensors artifacts are not numerically interchangeable.
On the same normalized random input their final feature maps had low cosine
similarity, so a decoder trained with one backend must be inferred with that
same backend and model identity. **Compatibility ONNX** is therefore the Mimics
default and locks the verified encoder to `256x256`. **Native PyTorch weights**
remain an explicit alternative backend whose local HuggingFace configuration
defaults to `224x224`; it is not presented as protocol-equivalent.

The filename `model.safetensors` is overloaded by the two systems. In this
repository it is a HuggingFace DINOv3 encoder checkpoint. In the installed
reference workflow, the file produced after training is the segmentation
decoder checkpoint while the frozen encoder remains `model.onnx`.

### Decoder

The decoder is implemented by `FrozenFeatureUNet2D`:

| Stage | Feature path | Raw-image skip |
|---|---:|---:|
| Bottleneck | `384 -> 256` | none |
| Up 1 | `256 -> 128` | pixel-unshuffle factor 8, 64 channels |
| Up 2 | `128 -> 64` | pixel-unshuffle factor 4, 16 channels |
| Up 3 | `64 -> 32` | pixel-unshuffle factor 2, 4 channels |
| Up 4 | `32 -> 16` | original grayscale, 1 channel |
| Output | `16 -> 2` | none |

Each double-convolution block uses two bias-free `3x3` convolutions,
affine instance normalization, leaky ReLU with slope `0.01`, and a residual
connection around the second convolution. The transpose convolutions use
kernel and stride 2. Weight initialization follows the verified tinygrad
uniform bound.

This is independent 2D decoding. Ordered slices are stacked to form the 3D
mask; there is no hidden 3D convolution or through-plane resampling.

### Optimization and validation

- Default optimizer: AdamW.
- Default learning rate: `5e-4`.
- Default weight decay: `1e-4`.
- Betas: `0.9`, `0.999`; epsilon: `1e-8`.
- Default loss: cross entropy.
- Default real slice batch: 4.
- Final incomplete batch is dropped only when at least one complete batch
  already exists.
- Each epoch reshuffles all slices across all selected cases.
- Independent height and width flips are sampled during training.
- Epoch cosine scheduling has a floor of 10% of the base learning rate.
- Validation runs at the selected interval and on the final epoch.
- Validation averages identity, height-flip, width-flip, and combined-flip
  probabilities.
- Validation loss is the mean of the four transformed losses.
- Dice is calculated on each complete 3D case and then averaged.
- Equal or improved validation Dice replaces the best checkpoint.

## Mimics Data Contract

This method deliberately avoids an additional training-space reorientation:

1. Mimics label export restores every mask to the original source-image
   shape and affine.
2. Dataset materialization pairs that source-grid label with the original
   source image.
3. Training validates image/label shape and affine before resizing slices.
4. Inference starts from the same original image and writes the prediction
   with the original shape, affine, qform, and sform.
5. The existing Mimics bridge maps that source-grid prediction to the open
   Mimics image grid when applying the result.

An affine mismatch is a hard error. Shape equality alone is not accepted as
proof that image and label occupy the same physical grid.

### Label sources

The source dataset remains the image source in every mode. Training labels have
three explicit sources:

1. **Refresh from saved .mcs** exports the current saved Mimics masks into a
   job-scoped staging directory before training.
2. **Use source dataset segmentations** reads labels already stored below each
   source case.
3. **Use an exported masks folder** reuses a previous Export Masks destination
   without opening Mimics or exporting again.

The reusable export layout may be either
`<label-root>/<case>/segmentations/<organ>.nii.gz` or
`<label-root>/<case>/<organ>.nii.gz`. Case identifiers still come from the
source image dataset, preventing a label-only folder from being mistaken for
an image dataset. Selecting an external label root disables fallback to source
dataset labels, so missing exported labels are reported rather than silently
mixing two label versions. The selected source and path are recorded in the job
status, `samples.json`, and the registered model manifest.

### Feature-cache lifecycle

Frozen Feature 2D is the only current training path that caches frozen encoder
features. The volume, 2D, and 2.5D fine-tuning paths continue to run their
encoder during training iterations.

Each case cache contains memory-mapped `embeddings.npy`, normalized
`images.npy`, `labels.npy`, and a manifest. The raw image slices remain
necessary for the decoder's image skip connections. Cache construction writes
to a `.building` directory and atomically publishes the completed cache, so a
cancelled or failed build cannot look complete.

The cache is private to one training job and is stored below:

`<workspace>/runs/<organ>/<job-id>/training_artifacts/<experiment>/feature_cache`

It is never reused across jobs, datasets, or encoder backends. This avoids stale
feature contamination. Normal completion, cancellation, and handled failures
close all memory maps before deleting the cache, which is required on Windows.
If the worker is force-killed, the next workspace maintenance pass detects the
dead controller and removes the recorded experiment directory. Cache retention
is disabled for Mimics-launched jobs.

## Mimics Workflow and Lifecycle

The training entry opens the external PySide6 setup window immediately. Dataset
selection, directory scanning, preflight, model loading, feature caching,
training, and inference remain outside the Mimics Python process. Dataset
scanning runs on a worker thread in that external process, and Start Training is
disabled until the scan completes. The default configuration does not fall back
to Mimics-native parameter dialogs when the external UI is unavailable; it
reports the missing offline UI dependency and starts no task. The legacy
internal fallback remains available only as an explicit configuration choice.

The setup window remains an external PySide6 process. Starting training
launches the pipeline in a separate Python process and returns control to
Mimics. Mimics does not run feature extraction, model loading, training, or
inference on its GUI thread.

The status JSON reports feature preparation by case and slice, then epoch,
training loss, validation loss, validation Dice, completion, cancellation,
or failure. The feature cache is job-scoped and deleted in `finally` unless
retention is explicitly enabled. Cancellation is checked during feature
preparation and every train/validation batch.

The UI displays this method as **Frozen Feature 2D**. Incompatible controls
are disabled or normalized before submission:

- the encoder is frozen;
- ViT-S/16 is required;
- sampling is full native axial slices;
- channel input is repeated grayscale;
- 3D patching, sub-volume training, mixed precision, and gradient
  accumulation are disabled;
- slice batch size remains configurable;
- the bundled default ONNX encoder locks image size to `256x256`;
- a custom dynamic-spatial ONNX encoder may use an explicitly configured
  multiple-of-16 size, including `512x512`;
- a fixed encoder and a conflicting configured size fail preflight instead of
  silently resizing to a different model contract;
- a custom encoder path may expose another size, which is validated against
  the model metadata before GPU acquisition.
- Native PyTorch weights use `model.safetensors`; the selected backend and
  checksum are saved with the model so inference restores the same encoder.

## Numerical Verification

The following checks were run locally on 2026-07-23/24.

### Reference component comparison

Using one SynthStrip volume:

- percentile normalization maximum absolute error: `0`;
- bilinear image resize maximum absolute error: `0`;
- nearest-neighbor label resize: exact pixel equality.

Using identical decoder weights and identical random tensors:

- PyTorch and reference decoder state keys and shapes: exact match;
- decoder maximum absolute output error: `6.85453415e-06`;
- decoder mean absolute output error: `1.09720747e-06`;
- outputs pass `atol=1e-4, rtol=1e-4`.

### End-to-end SynthStrip check

Training data:

- train: `asl_epi_101`, `asl_epi_106`;
- independent validation: `asl_epi_109`;
- native shape for each checked case: `64x64x22`;
- training cases: `asl_epi_101` and `asl_epi_106` (44 axial slices);
- validation case: `asl_epi_109` (22 axial slices);
- 8 epochs, batch 4, cross entropy, AdamW, learning rate `5e-4`.
- 88 decoder optimization batches after the 66 slices were encoded once;
- 2,126,578 trainable decoder parameters; the ONNX encoder remained frozen;
- measured training wall time: 68.12 seconds on this small smoke dataset.

Observed results:

- train loss: `0.3669 -> 0.1219`;
- best validation Dice: `0.9517` at epoch 5;
- independent inference Dice: `0.9525129673`;
- output shape equals source shape;
- output affine equals source affine;
- qform and sform codes are both valid;
- the transfer optimization before/after predictions differ by zero voxels.

The same three cases were also run through the native PyTorch safetensors
backend at 224:

- 2 training epochs, batch 4, cross entropy, AdamW, learning rate `5e-4`;
- train loss: `0.3662 -> 0.2166`;
- best validation Dice: `0.9390`;
- independent restored-grid inference Dice: `0.9406031342`;
- output shape: `64x64x22`.

This is a training/save/load/inference smoke validation, not evidence that the
safetensors and ONNX encoders are equivalent.

The local regression suite passed 113 focused model/research tests. The
repository-wide suite exercises 355 import, export, lifecycle, UI, model, and
geometry tests.

## Windows GPU Deployment

The target environment is Windows x64 with Python 3.13 and NVIDIA CUDA.

- `onnxruntime-gpu==1.22.0` is the pinned offline runtime.
- The matching `cp313-win_amd64` wheel is included by the offline bundle
  builder.
- The setup worker treats `onnxruntime` as a required import.
- On Windows, ONNX Runtime preloads CUDA/cuDNN DLLs before session creation
  when the API is available.
- If PyTorch sees CUDA but ONNX Runtime does not expose
  `CUDAExecutionProvider`, training fails during preflight with an actionable
  environment error.
- The compatibility encoder must be provided at
  `external/dinov3-medical-seg/models/dinov3-vits16/model.onnx`. Native PyTorch
  mode uses `model.safetensors` plus the HuggingFace model configuration from
  the same directory. An equivalent explicit model directory may be selected.
  Model binaries are not stored in Git.
- Portable-package and Windows setup checks now fail immediately when the
  default encoder is absent.

Windows acceptance still requires one real GPU run to record:

1. CUDA provider activation;
2. feature-cache throughput and peak GPU memory on a 3060-class card;
3. one training epoch and one restored-grid inference;
4. nonblocking Mimics interaction while the external process runs;
5. cancellation during feature preparation and during training;
6. cleanup of the job feature cache, process, GPU allocation, and status
   terminal state.

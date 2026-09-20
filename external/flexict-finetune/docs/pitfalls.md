# Pitfalls & Lessons Learned

The recipe in this repo is the survivor of many failures. Each issue below was
diagnosed and fixed during development; they are encoded into the trainer /
scripts so you do not hit them again. Documented here so the reasoning survives.

## 1. FlexiCT RoPE overflows in fp16 → must train in fp32

**Symptom:** loss becomes NaN early in training; EMA pseudo-dice collapses.

**Root cause:** FlexiCT's RoPE (rotary position embedding) position encoding
produces values that overflow fp16 range under autocast. The NaN appears in the
backward pass first.

**Fix (`FlexiCTFP32Trainer`):** set `self.grad_scaler = None` (disables
`GradScaler`) and override `train_step` to run the forward inside
`dummy_context()` (a no-op) instead of `autocast(enabled=True)`. Pure fp32
forward + backward. Validation keeps the default path (fp16 forward-only is
fine — the NaN only shows up in backward).

Do NOT re-enable fp16/autocast "to go faster" — it will NaN within a few
epochs.

## 2. Mirroring the lateral axis flips a single-side target

**Symptom:** predictions show a false-positive "ghost" on the contralateral
side (e.g. a left-only target predicts a right-side blob too).

**Root cause:** nnU-Net's default `mirror_axes` includes the lateral (L-R) axis.
Mirroring that axis turns a left-side target into a right-side one during
augmentation, so the model learns both sides and predicts on both at inference.

**Fix:** `MIRROR_DISABLE_AXES`. The trainer defaults to nnU-Net's standard
mirroring (2D `(0,)`, 3D `(0,1,2)`), but if your target is single-side /
asymmetric, export `MIRROR_DISABLE_AXES=1` (the L-R axis after nnU-Net's
transpose) to exclude it. Inference TTA mirroring follows the same axes, and
`run_predict.sh` passes `--disable_tta` to be safe.

For symmetric / midline targets (e.g. liver), leave mirroring at default.

## 3. Elastic deformation: a silent data/segmentation desync

**Symptom:** "ghost organ" oversegmentation that looks like overfitting but is
actually corrupt batches; pseudo-dice looks artificially high then collapses on
real validation.

**Root cause:** the `batchgeneratorsv2` `SpatialTransform` elastic API was
invoked with `elastic_deform_scale` as a *range* where it expects a *per-axis
tuple* (len-3 for 3D). The resulting `IndexError` is swallowed inside a worker
process, which then returns a mis-aligned data/segmentation pair. Training
proceeds on garbage; the desync inflates pseudo-dice and destroys real
generalization.

**Fix:** elastic is left at `p_elastic_deform=0` (off). The rest of nnU-Net's
augmentation (rotation, scaling, brightness, contrast, gamma, gaussian
noise/blur, simulate-lowres) is kept at nnU-Net default — do NOT crank
intensity augmentation either (see #4).

## 4. Strong intensity augmentation teaches a "bright = foreground" shortcut

**Symptom:** after strengthening brightness/contrast/gamma to fight few-shot
overfitting, the model predicts any bright structure as foreground — ghost
blobs on bones/contrast-filled vessels, not the target.

**Root cause:** aggressive intensity augmentation under few-shot data makes
"high intensity" the cheapest foreground cue. The model latches onto it
instead of target shape/anatomy.

**Fix:** keep nnU-Net's *default* intensity augmentation. Do not widen the
ranges. The validated recipe uses the default `get_training_transforms` —
strength comes from the pretrained FlexiCT backbone + the multi-scale decoder,
not from augmentation severity.

## 5. Two-LR optimizer: backbone ≠ decoder

**Symptom:** few-shot finetuning is unstable if the whole network uses one LR:
either the pretrained backbone drifts too fast (destroying the prior) or the
randomly-init decoder learns too slowly.

**Fix (`configure_optimizers`):** two AdamW param groups — backbone
(`vit_lr=3e-5`, matched by name fragments `dino_encoder`/`backbone`/`vit.`) and
decoder (`initial_lr=3e-4`, everything else). 10× ratio. Poly schedule
`(1-progress)^1.0`. `betas=(0.9, 0.98)`.

## 6. Multi-scale decoder: wide concat beats 1x1x1 projection

**Symptom:** a 1x1x1 conv projection (3456→864) before the decoder, added to
"match the paper", dropped few-shot Dice substantially vs no projection.

**Root cause:** under few-shot, the 4× channel bandwidth of a direct concat
(n_levels × embed_dim) retains multi-scale detail the projection bottleneck
throws away.

**Fix:** `FlexiCTPrimus2D`/`FlexiCTPrimus3D` concatenate the 4 levels directly
into the patch-decode ladder, no projection. Do not add a projection "to be
tidy" — it costs Dice. (The projection variant is intentionally NOT in this
repo.)

## 7. 3D decoder upsampling: trilinear+conv, not ConvTranspose3d

**Why:** `ConvTranspose3d`'s backward allocates a huge intermediate activation
that OOMs at practical patch sizes. The 3D patch-decode ladder uses
`interpolate(trilinear)` + `Conv3d` per stage instead — bounded memory, same
upsampling factor. This is the validated 3D decoder; do not swap to
ConvTranspose3d without checking memory.

## 8. Lower batch_size in the PLANS, never in an initialize() override

**Symptom:** 2D smoke OOM'd at 44 GB even after "clamping" batch_size 49→8 in a
trainer `initialize()` override. A direct forward+backward at bs=8 used only
14 GB, so the OOM was bogus.

**Root cause:** nnU-Net's `_set_batch_size_and_oversample` runs *inside*
`super().initialize()` and builds the oversample sampling schedule from the
plans' batch_size (49). Clamping `self.batch_size` to 8 *after* super() returns
does NOT rebuild that schedule, so the dataloader still produced 49-sample
batches → forward at bs=49 → OOM.

**Fix:** do NOT override `initialize()` to change batch_size. Instead, edit the
batch_size directly in `nnUNetPlans.json` after preprocessing, so nnU-Net's
native flow sees the right value end-to-end. Use
`scripts/set_plan_batch_size.py`:

```bash
python scripts/set_plan_batch_size.py --preprocessed $nnUNet_preprocessed \
    --dataset 907 --config 2d --batch-size 8
```

Validated memory (A40 46GB, fp32): 2D bs=8 patch256² ≈ 5 GB in training
(14 GB for raw fwd+bwd); 3D bs=2 patch128³ ≈ 32 GB.


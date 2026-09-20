# ScribblePrompt In Mimics

## Entry

**ScribblePrompt** accepts positive/negative clicks, positive/negative
scribbles, and one foreground box on one axial, coronal, or sagittal slice.
The selected Mask may be empty or may provide the starting segmentation.

The entry returns to Mimics immediately after the necessary native prompt or
Mask-buffer collection. Computation runs under `nninteractive_env` in a hidden,
below-normal-priority process. Reopening the same entry shows progress and a
Stop action. Phase changes and failures are also written to the Mimics log.

## ScribblePrompt Contract

The integration uses the official ScribblePrompt-UNet architecture and 128 x
128 input contract. The active grayscale slice is min-max normalized to
`[0,1]`. The five model channels are image, box, positive click/scribble,
negative click/scribble, and previous logits/Mask input. Only the prompted
slice is replaced in the 3D Mask.

All prompts in one prediction must lie on the same slice. A box or scribble
identifies that slice directly. For an initial click-only prediction, the user
chooses the active axial, coronal, or sagittal view; the integration maps that
view to a Mimics buffer axis from the image voxel-to-RAS matrix instead of
assuming fixed array axes.

When a prior ScribblePrompt result is unchanged, the next invocation reuses its
saved logits. If the Mask has been edited, stale logits are rejected and the
selected Mask is converted to a bounded logit prior. This preserves manual
edits and prevents an old AI state from silently replacing them.

An empty Mask needs at least one foreground click, foreground scribble, or
box. Once a Mask or previous prediction exists, a background-only correction
is valid and can remove an over-segmented region.

The official checkpoint is not stored in Git. Place it at:

`external/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt`

## Result And Recovery

Completion presents one decision instead of a separate notification:

- **Update Selected Mask**
- **Create Editable Copy**
- **Discard**

If the selected Mask changes or is deleted while the worker is running,
in-place update is disabled and only a copy can be created. Temporary prompt
Masks are removed on completion, failure, cancellation, and global background
service cleanup.

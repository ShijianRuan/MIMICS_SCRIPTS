# ITK Snake And ScribblePrompt In Mimics

## Entries

- **ITK Snake** runs a 3D ITK geodesic active contour from the selected,
  non-empty Mask.
- **ScribblePrompt** accepts foreground and optional background scribbles on
  one axial, coronal, or sagittal slice. The selected Mask may be empty or may
  provide the starting segmentation.

Both entries return to Mimics immediately after the necessary native prompt or
Mask-buffer collection. Computation runs under `nninteractive_env` in a hidden,
below-normal-priority process. Reopening the same entry shows progress and a
Stop action. Phase changes and failures are also written to the Mimics log.

## ITK Snake Safety Contract

The selected Mask is the initial contour; ITK Snake does not segment an organ
from an empty image. Image and Mask buffers use the same Mimics grid. The
worker explicitly converts Mimics `x,y,z` array order to SimpleITK `z,y,x`
array order while retaining physical voxel spacing.

Contour changes are restricted to a configurable physical-distance band around
the original boundary. Empty output and excessive volume changes fail closed,
leaving the project unchanged. Conservative, Balanced, and Aggressive change
the permitted displacement and evolution settings; they are safety profiles,
not organ presets.

## ScribblePrompt Contract

The integration uses the official ScribblePrompt-UNet architecture and 128 x
128 input contract. The active grayscale slice is min-max normalized to
`[0,1]`; foreground and background scribbles occupy their official two prompt
channels. Only the prompted slice is replaced in the 3D Mask.

When a prior ScribblePrompt result is unchanged, the next invocation reuses its
saved logits. If the Mask has been edited, stale logits are rejected and the
selected Mask is converted to a bounded logit prior. This preserves manual
edits and prevents an old AI state from silently replacing them.

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

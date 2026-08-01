# IGAC In Mimics

## User Workflow

1. Open an image and select exactly one Mask in Mimics. The Mask may be empty
   for a small image, but a rough seed is recommended for normal CT/MR volumes.
2. Run `02_AI/IGAC.py`. Mimics takes one image/Mask snapshot and opens an
   independent PySide6 window.
3. Inspect the contour in axial, coronal, and sagittal views. **Mimics** restores
   the contrast captured when the workspace opened; **Auto** uses a robust image
   range. Width and level affect display only, never contour evolution. Use **Add** to
   force foreground, **Barrier** to protect background, and **Neutral** to
   remove local guidance. Drawing temporarily pauses evolution and then resumes
   it. Dragged strokes are continuous physical 3D capsules rather than a series
   of disconnected event points.
   A non-empty selected Mask is centered automatically on its middle slice and
   enlarged with 12 mm of context. **Fit Mask**, **Fit Image**, zoom buttons,
   Ctrl+wheel zoom, and middle-button pan do not alter voxel coordinates.
4. Choose **Update selected** or **Editable copy**, then select **Apply to Mimics**.
   Closing or cancelling the window leaves Mimics unchanged.

If the selected Mask changes in Mimics while IGAC is open, the completed result
is always applied to a new editable copy. It never overwrites newer work.

## Architecture

- `scripting_library/02_AI/IGAC.py` is the flat Mimics entry.
- `runtime_py35/interactive_algorithms_mimics.py` performs the one-time buffer
  snapshot in bounded chunks, starts the visible process, monitors status,
  handles stop requests, checks for concurrent Mask changes, and performs the
  final buffer update. The image is not hashed because it is immutable during
  the session; the Mask is hashed for conflict protection.
- `tools/igac_gui.py` owns the non-modal PySide6 window. A `QThread` owns all
  contour computation; paint, resize, navigation, and button events remain on
  the GUI thread.
- `tools/igac_engine.py` implements the 3D LGDF-style contour in PyTorch. It
  keeps the exact Mimics `(x, y, z)` grid and uses physical spacing for Gaussian
  neighbourhoods, derivatives, curvature, signed distance, and brush radius.

PySide6 is sufficient for the external interaction layer. It is not the compute
engine. The implementation uses the existing PyTorch/CUDA runtime in
`nninteractive_env`, so a separate OpenCL SDK, compiler, or runtime is not
required. CPU fallback is available only for bounded workspaces.

## Resource And Failure Behaviour

- A CUDA session owns the project-wide `gpu.lock` until Apply, Cancel, failure,
  or window close. DINOv3, nnU-Net, nnInteractive, and IGAC therefore cannot
  overcommit the same GPU.
- The external window remains responsive while waiting for the GPU or evolving
  the contour. The Mimics process never runs the iterative algorithm.
- Mimics GUI refreshes between snapshot chunks. The final result uses one atomic
  `set_voxel_buffer()` operation so a partial Mask can never appear in the
  project. That final API call may still cause a brief handoff pause on very
  large volumes and must be measured in the target Mimics build.
- Closing Mimics, using **Stop All Owned Services**, or creating the job cancel
  marker stops the worker and releases the lock.
- Status and English diagnostics are stored under
  `.mimics_runtime/interactive_algorithms/igac/<job>/` and terminal jobs follow
  the shared retention policy.
- A non-empty initial Mask creates a physical-margin ROI. Empty Masks use the
  full image only below the configured voxel limit; larger images require a
  seed to prevent GPU/CPU memory exhaustion.
- **Limit auto movement** keeps autonomous evolution within a physical shell
  around the initial Mask. Add and Barrier are symmetric hard constraints and
  intentionally override that shell, so an annotator can still correct a true
  boundary beyond the automatic limit.

## Parameters

The GUI intentionally exposes only the controls needed during interaction:

| Control | Meaning |
| --- | --- |
| Range | Physical Gaussian scale of the local intensity model, in mm. |
| Smooth | Curvature regularity weight. |
| Grow | Small foreground/background bias; positive values favour growth. |
| Limit auto movement | Maximum inward/outward automatic displacement from the initial Mask, in mm. |
| Brush size | Physical diameter of Add/Barrier/Neutral guidance, in mm. |
| Mimics / Auto | Restore Mimics contrast or use the image's robust range. |
| Width / Level | Display-only GV contrast for both CT and MR. |
| Fit Mask | Center the local organ region with 12 mm physical context. |
| Fit Image | Return to the complete slice. |
| − / + | Zoom while preserving full-image prompt coordinates. |

The workspace reads the exact Mimics image voxel buffer. These values are
Mimics Gray Values: CT commonly has a linear HU-to-GV mapping, while MR has no
HU meaning. The 3D engine performs one robust linear normalisation for its local
statistics, so an additive HU/GV offset does not change the model. Display
windowing is kept separate from that algorithm normalisation.

Numerical safeguards and deployment limits remain in
`interactive_algorithms_config.json` rather than burdening the annotator.
The default Grow value is neutral (`0.0`) and the automatic movement limit is
10 mm.

## Performance Expectation

One IGAC iteration performs six separable 3D Gaussian operations, equivalent to
18 one-dimensional `conv3d` passes, in addition to curvature and regularisation.
The expected interaction is therefore animated, seconds-scale convergence, not
an instantaneous snap. A 5-10 million voxel ROI may require tens of seconds for
hundreds of iterations depending on the GPU. The GUI reports actual iterations
per second. `iterations_per_cycle` can reduce frame/status overhead, but larger
values also delay brush-command handling; ROI reduction or a separately
validated coarse-to-fine engine is required for a larger speedup.

## Compatibility And Validation

The implementation follows the interactive 3D LGDF design and tool semantics
described by the [IGAC project](https://github.com/cwkx/IGAC/) and its published
method. It is a PyTorch reimplementation for Mimics, not a copy of the upstream
OpenCL/OpenGL UI. The physical-spacing correction is intentional: the upstream
voxel-unit kernels do not represent anisotropic clinical spacing correctly.

Automated validation covers anisotropic spacing, all three view mappings,
continuous 3D brush strokes, symmetric hard Add/Barrier guidance, the physical
movement shell and its manual-override precedence, finite contour evolution,
hard-boundary sphere convergence without leakage, result-grid preservation,
zoomed-canvas-to-full-grid coordinate mapping, display-window isolation from
the 3D result, chunked buffer transfer,
external destination choice, concurrent Mask-change protection, cancellation,
and terminal status creation. Real Mimics validation is still required for the
duration of the initial `get_voxel_buffer()` snapshot and the final
`set_voxel_buffer()` call on representative large Windows datasets.

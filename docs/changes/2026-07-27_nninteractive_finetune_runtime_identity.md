# nnInteractive Fine-Tuning Runtime Identity Fix

Date: 2026-07-27

## Problem

Legacy task-model registration changed only selected checkpoint dictionary keys.
The nnInteractive network exposes some shared parameters through multiple
state-dict aliases. During `load_state_dict`, a later unchanged alias could
overwrite the fine-tuned value. The checkpoint file checksum changed, but the
effective network reconstructed for inference could still equal the official
model.

The same workflow also lacked a clear choice when several models existed for
one task, exposed prompt types that had not been validated during training, and
reported 100% before model comparison and registration were complete.

## Changes

- Export the complete network state and check shared parameter aliases before
  publishing a checkpoint.
- Reconstruct a fresh nnInteractive network after training memory is released
  and compare its effective parameter fingerprint with the trained network.
- Store runtime verification, effective fingerprint, checkpoint checksum, and
  validated prompt capabilities in model metadata.
- Verify the registered checkpoint checksum when starting or switching the
  background inference server. Reused prompt calls do not re-hash the file.
- Reject incomplete, corrupt, or runtime-identity-mismatched task models and
  portable model packages.
- Preserve the model attached to an existing Mask. For ambiguous new work,
  show the external task/model chooser and support a project-specific binding.
- Show model IDs in Model Versions and replace ambiguous `Use` actions with
  `Set Active` or `Annotate with This Model`.
- Limit click-trained task models to Point prompts. The official model retains
  Point, Scribble, Box, and Lasso support.
- Use modality-neutral `CLoPA-IN` and `CLoPA-CN` names for CT and MR.
- Reserve progress ranges for preparation, training, runtime verification,
  model comparison, and registration. Only the controller terminal state can
  reach 100%.
- Refresh task and model history immediately after a terminal job transition.

## Legacy Model Migration

`tools/register_brain_model.py` creates an immutable repaired model version.
It prefers the validated NPZ weights when available. Otherwise, it recovers the
canonical CLoPA-IN values directly from the legacy checkpoint, writes a complete
state dict, performs a fresh-network reload check, marks the old version
corrupt, and selects the repaired version.

Model checkpoints and the local task-model registry remain ignored by Git and
must be migrated or transferred separately on the Windows workstation.

## Validation

- Fine-tuning package: 18 tests passed.
- nnInteractive defensive integration: 73 tests passed.
- Fake Mimics workflows: 14 tests passed.
- Project comprehensive test suite completed successfully.
- PySide6 Model Center and model chooser instantiated successfully in an
  offscreen layout check.
- The repaired brain model reloaded with the same effective fingerprint as the
  exported network and a different fingerprint from the legacy effective
  network.

Runtime identity is verified, but segmentation quality remains a separate
acceptance criterion. The available brain manifest contains 48 training cases
and 2 validation cases, so independent held-out Mimics validation is still
required before declaring the model clinically useful.

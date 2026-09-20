"""Per-voxel segmentation uncertainty + case ranking.

Self-contained (numpy / scipy / nibabel / matplotlib only — no nnU-Net or torch
dependency). Computes voxel-wise disagreement across multiple algorithm masks of
the same case, saves per-case uncertainty NIfTI maps, and ranks all cases by an
aggregate uncertainty metric (integrated uncertainty = Σ voxel-uncertainty ×
voxel volume, which rewards both *high* and *large* disagreement regions).

This is the generic Action6 method, integrated into the FlexiCT framework so the
trained 2D and 3D models can serve as the two (or more) algorithms: run each
model on the same inputs, point ``run_batch`` at the prediction directories, and
the cases where the two models disagree most bubble to the top of the ranking.

Typical entry points:
    run_batch(mask_dirs, out_dir, ...)          — compute + save per-case maps
    analyze_uncertainty_dir(out_dir, ...)       — read maps back, rank, write CSV
"""

from .uncertainty import (  # noqa: F401
    compute_uncertainty,
    compute_uncertainty_independent,
    compute_case_metrics,
    run_case,
    run_batch,
    analyze_uncertainty_dir,
    read_case_list,
    resolve_mask_paths,
)

__all__ = [
    "compute_uncertainty",
    "compute_uncertainty_independent",
    "compute_case_metrics",
    "run_case",
    "run_batch",
    "analyze_uncertainty_dir",
    "read_case_list",
    "resolve_mask_paths",
]

# Uncertainty Ranking (2D vs 3D model disagreement)

This integrates the Action6 per-voxel uncertainty method (`uncertainty/uncertainty.py`)
into the FlexiCT framework. The trained 2D and 3D models serve as the two
"algorithms": run both on the same inputs, compute per-voxel disagreement, and
rank every case by an aggregate uncertainty metric. The cases at the top of the
ranking are where the two models disagree most — the natural next targets for
review, re-annotation, or additional few-shot training data.

## Method

For each case, the N model masks (here N=2: 2D + 3D) are stacked. At every
voxel, the masks either agree or split. Action6 turns that into a per-voxel
uncertainty in `[0,1]` via one of:

| method | per-voxel definition | N=2 values |
|---|---|---|
| `disagreement` | `1 − max_vote_count / N` | agree→0, split→0.5 |
| `variance` | `p(1−p)`, p = foreground fraction | agree→0, split→0.25 |
| `entropy` | normalized vote entropy `/ log(K)` | agree→0, split→1 |
| `compare` | `d·a` vs the first mask as baseline | agree→0, baseline-wrong→up to 1 |

Each label is handled one-vs-rest, then per-voxel max combines them into one
map (only relevant for multi-label targets; liver is a single label, so this is
trivial). The map is saved as a NIfTI (scaled to uint8).

The ranking key is **integrated uncertainty** = `Σ voxel_uncertainty × voxel
volume (mm³)`, which rewards both *high* and *large* disagreement regions. With
`disagreement` and N=2 this reduces to **(disagreement volume) × 0.5**, i.e. the
total volume where the 2D and 3D predictions differ — intuitive and well-scaled.

## Why `disagreement` for 2 masks

With only 2 masks, `entropy` saturates at a fixed max (1.0 at every split voxel),
so the integrated value is just `0.5 × split_count × ... ` up to a constant —
still rankable but no richer than `disagreement`. `disagreement` makes the
"fraction disagreeing" semantics explicit (0 / 0.5 per voxel). `variance`
(0 / 0.25) is equivalent up to a scale. `compare` adds a baseline-reference
twist (treats `preds_2d` as the reference). `disagreement` is the default; all
four are selectable via `--method`.

## Usage

### One-command pipeline (remote)

```bash
# prepare inputs -> predict 2D (GPU0) + 3D (GPU2) concurrently -> rank
bash scripts/run_uncertainty_rank.sh <source> 907 0 0 2   # full (all cases)
bash scripts/run_uncertainty_rank.sh <source> 907 5 0 2   # smoke (5 cases)
```

### Step by step

```bash
# 1. link source CTs into nnU-Net input names (<case>_0000.nii.gz) + case list
python scripts/prepare_inputs.py \
    --source <v201_root> --out infer_input --cases-file infer_input/case_list.txt

# 2. predict with both models (2D fast, 3D slow — run on separate GPUs)
bash scripts/run_predict.sh 907 2d         infer_input preds_2d
bash scripts/run_predict.sh 907 3d_fullres infer_input preds_3d_fullres

# 3. rank by disagreement
python scripts/run_uncertainty.py \
    --mask-dirs preds_2d preds_3d_fullres \
    --cases-file infer_input/case_list.txt \
    --out uncertainty_out \
    --method disagreement \
    --filename-template "{case}_0000.nii.gz" \
    --sort-key integrated \
    --target-labels 1
```

## Output

- `uncertainty_ranking.csv` — one row per case, sorted high→low by
  `integrated`. Columns: `rank, case, integrated, uncertain_vol, mean,
  mean_nonzero, max, n_uncertain, cc_integrated, cc_vol, ...` (and per-level
  volumes). The top rows are the cases with the largest 2D-vs-3D disagreement.
- `*_uncertainty_disagreement.nii.gz` — per-voxel uncertainty map (uint8,
  scaled ×10), one per case.
- `*_consensus.nii.gz` — majority-vote mask (unless `--no-consensus`).

## Notes

- **Filename alignment.** nnU-Net predict names outputs after the input
  (`<case>_0000.nii.gz`), so `--filename-template "{case}_0000.nii.gz"` matches.
  If you rename predictions to `<case>.nii.gz`, use `"{case}.nii.gz"`.
- **Single binary target.** With `--target-labels 1` there is no label interface,
  so the `partition_boundary` component pass is skipped automatically.
- **Reorientation.** The dataset uses `NibabelIOWithReorient`; the nnU-Net
  predictor reorients each input at read time, so `prepare_inputs.py` only
  normalizes the filename suffix, not the orientation.
- **Cost.** 3D inference is the bottleneck (~1–3 min/case); 1228 cases is a
  multi-day run. 2D is ~10× faster. Run them concurrently on separate GPUs.

## Validation (28-case test set)

Smoke-validated on the 28-case Dataset907 test set (2D + 3D predictions already
on disk). `run_uncertainty.py --method disagreement --target-labels 1` produced
`uncertainty_ranking.csv` with all 28 cases ranked by integrated uncertainty.

The ranking is sanity-correct:
- **Rank 1 = s0623** (integrated 1.18M, 701k disagreement voxels) — this is the
  known 3D outlier (3D Dice 0.54 / HD95 144 mm), i.e. the case where 2D and 3D
  disagree most. The ranking surfaces it as #1, exactly as expected.
- **Ranks 23–28 = 0** (s0063, s0292, s0715, s0925, s1072, s1260) — these are the
  cases where 2D and 3D agree perfectly (they are also the Dice-1.0 cases in the
  evaluation), so zero disagreement.
- `mean_nonzero` = 0.5 and `max` = 0.5 everywhere non-zero — the exact 2-mask
  signature (each split voxel has the two masks 50/50).

So with 2 masks, `disagreement` + `integrated` reduces to **total disagreement
volume**, and the ranking correctly orders cases by how much 2D and 3D differ.

## Full 1227-case run (completed)

The full v201 run completed via streaming (1227 cases with ct+liver; 1 case in
v201 has no liver label and was skipped). 2D and 3D FlexiCT models predicted
each case; per-voxel disagreement ranked all cases.

**Result summary** (`unc_out/uncertainty_ranking.csv`):
- 958 / 1227 cases have disagreement (2D and 3D differ somewhere)
- 269 / 1227 cases are fully consistent (2D == 3D everywhere, integrated=0)
- Top 10 most-disagreeing cases: s0623 (1.18M), s0014 (1.04M), s0253 (0.93M),
  s0941 (0.77M), s0589 (0.60M), s0979 (0.46M), s1412 (0.44M), s0658 (0.44M),
  s0543 (0.41M), s0593 (0.37M) — integrated = 0.5 × disagreement-volume.
- rank 1 = s0623, the same case that was the known 3D outlier in the 28-case
  test set (3D Dice 0.54 / HD95 144 mm) — the ranking correctly surfaces it as
  where the two models disagree most.

These top cases are the natural next targets for review / re-annotation /
additional few-shot training data — they are where the 2D and 3D models split.

The run used ``scripts/stream_uncertainty_rank.py`` (local-driven streaming:
each CT scp'd to the remote GPU box, predicted with 2D+3D, scp'd back, deleted
on the remote — so remote disk only ever holds one case; predictions accumulate
locally; ranking runs locally). Per-case caching makes it survive interruptions
(the background task is killed on context compaction; on resume, cached cases
are skipped). ``remote_predict_one.sh`` runs 2D then 3D serially under a flock
(parallel was tried but caused a stage-dir race). If the data is already on the
GPU box, ``scripts/run_uncertainty_rank.sh`` is the non-streaming alternative.
``scripts/run_uncertainty_rank.sh`` is the non-streaming alternative.


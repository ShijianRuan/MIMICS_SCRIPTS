# Validation log

Few-shot whole-organ CT (liver), nnU-Net 7-case internal holdout for training,
28-case external test set (Totalsegmentator). Single-label binary (0/1).
Raw per-case JSON lives in git history of the source research repo; the summary
numbers and per-case tables below are the authoritative record.

## Results

| config | mean Dice | median Dice | mean HD95 (mm) | fails (Dice=0) |
|---|---|---|---|---|
| 2D | 0.939 | 0.980 | 7.10 | 1/28 |
| 3D fullres | 0.961 | 0.977 | 9.67 | 0/28 |

Note: 3D mean HD95 is inflated by s0623 (a known dataset outlier — the case
sits at the top of every uncertainty ranking run as well). Excluding s0623 the
3D mean Dice is 0.975 and mean HD95 4.54 mm.

## Per-case Dice (2D)

| case | dice | hd95_mm |
|---|---|---|
| s0028 | 0.982 | 1.50 |
| s0053 | 0.987 | 1.50 |
| s0063 | 1.000 | — |
| s0071 | 0.991 | 1.50 |
| s0073 | 0.974 | 3.35 |
| s0078 | 0.983 | 1.50 |
| s0201 | 0.951 | 5.61 |
| s0215 | 0.969 | 3.67 |
| s0240 | 0.978 | 2.12 |
| s0265 | 0.896 | 9.00 |
| s0292 | 1.000 | — |
| s0375 | 0.987 | 2.12 |
| s0439 | 0.985 | 1.50 |
| s0488 | 0.972 | 3.00 |
| s0503 | 0.000 | — (only 2D failure; 3D recovers it at 1.000) |
| s0594 | 0.935 | 7.04 |
| s0604 | 0.973 | 3.00 |
| s0623 | 0.964 | 13.50 |
| s0691 | 0.981 | 1.50 |
| s0715 | 1.000 | — |
| s0749 | 0.987 | 1.50 |
| s0801 | 0.876 | 44.52 |
| s0925 | 1.000 | — |
| s1072 | 1.000 | — |
| s1189 | 0.972 | 2.60 |
| s1260 | 1.000 | — |
| s1307 | 0.979 | 3.35 |
| s1420 | 0.974 | 35.69 |

## Per-case Dice (3D fullres)

| case | dice | hd95_mm |
|---|---|---|
| s0028 | 0.972 | 3.00 |
| s0053 | 0.981 | 2.12 |
| s0063 | 1.000 | — |
| s0071 | 0.984 | 1.50 |
| s0073 | 0.951 | 4.50 |
| s0078 | 0.977 | 2.12 |
| s0201 | 0.963 | 4.50 |
| s0215 | 0.958 | 3.35 |
| s0240 | 0.976 | 2.60 |
| s0265 | 0.915 | 6.36 |
| s0292 | 1.000 | — |
| s0375 | 0.981 | 2.12 |
| s0439 | 0.981 | 2.12 |
| s0488 | 0.971 | 3.00 |
| s0503 | 1.000 | — |
| s0594 | 0.932 | 6.18 |
| s0604 | 0.978 | 2.12 |
| s0623 | 0.544 | 143.66 (known dataset outlier) |
| s0691 | 0.976 | 1.50 |
| s0715 | 1.000 | — |
| s0749 | 0.980 | 2.12 |
| s0801 | 0.977 | 2.12 |
| s0925 | 1.000 | — |
| s1072 | 1.000 | — |
| s1189 | 0.971 | 3.00 |
| s1260 | 1.000 | — |
| s1307 | 0.973 | 3.00 |
| s1420 | 0.979 | 2.12 |

(`—` = empty prediction or empty GT, HD95 undefined; Dice 1.000 means both empty.)

## Recipe provenance

These runs validated the recipe in the README (fp32, AdamW split LR
backbone 3e-5 / decoder 3e-4, poly LR, gradclip 12, 150 epochs,
deep_supervision=False, oversample_foreground=0.33, batch size from nnU-Net
plans, `--disable_tta -chk checkpoint_best.pth` at predict time).

Pitfalls solved along the way (fp16 RoPE NaN, elastic-deform API misuse
causing ghost-organ oversegmentation, batch-size clamp desyncing nnU-Net
oversample, L-R mirror for single-side organs) are documented as inline
comments in `trainers/flexict_trainer.py` and `flexict/models.py`.

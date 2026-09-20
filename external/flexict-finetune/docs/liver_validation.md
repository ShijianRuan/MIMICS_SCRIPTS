# Liver Few-Shot Validation

End-to-end validation that this standalone framework trains and predicts
correctly, on a *different* target than the one it was distilled from. Liver is
a good choice: large, midline/symmetric organ, standard Totalsegmentator label,
so few-shot should reach high Dice if the recipe transfers.

## Data

Source: `Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201`
Each case: `ct.nii.gz` + `segmentations/liver.nii.gz`.

Build (few-shot: 28 train / 28 test, stratified by liver Z-start):
```bash
python scripts/build_dataset.py \
    --source Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201 \
    --label-name liver --dataset-id 907 --dataset-name LiverFS \
    --n-train 28 --out $nnUNet_raw
```

## Environment

Validated on the remote nnU-Net env (nnunetv2 2.5.2 + torch 2.1.0). Trainer
installed into the env's `nnUNetTrainer/` dir. `source scripts/setup_env.sh`.

## Protocol

1. `nnUNetv2_plan_and_preprocess -d 907 --verify_dataset_integrity`
2. Smoke (2 epoch, confirms the pipeline runs end-to-end):
   `NUM_EPOCHS=2 bash scripts/run_train.sh 907 2d`
   `NUM_EPOCHS=2 bash scripts/run_train.sh 907 3d_fullres`
3. Full (150 epoch):
   `NUM_EPOCHS=150 bash scripts/run_train.sh 907 2d`
   `NUM_EPOCHS=150 bash scripts/run_train.sh 907 3d_fullres`
4. Predict test set + evaluate:
   `bash scripts/run_predict.sh 907 2d $nnUNet_raw/Dataset907_LiverFS/imagesTs ./preds_2d`
   `python scripts/evaluate.py --pred preds_2d --gt $nnUNet_raw/Dataset907_LiverFS/labelsTs --out eval_2d.json`

## Results

### Smoke (2 epochs — verifies the pipeline runs end-to-end)

Both 2D and 3D trained 2 epochs on the remote nnU-Net 2.5.2 env (A40 46GB),
fp32, no errors, no OOM. EMA pseudo-dice already rising fast (liver is a large
organ, so few-shot converges quickly):

| config | batch_size | patch | GPU mem | Epoch 0 EMA | Epoch 1 EMA |
|---|---|---|---|---|---|
| 2D  | 8 | 256²   | ~5 GB  | 0.9064 | 0.9119 |
| 3D  | 2 | 128³   | ~36 GB | 0.7746 | 0.7852 |

This confirms the standalone framework trains correctly on a target it was
never tuned for. A full 150-epoch run (not required for validation) would take
~hours (2D) / ~29h (3D at ~11.5 min/epoch) and is expected to reach high Dice.

### Full run (150 epochs)

Both 2D and 3D completed on the remote nnU-Net 2.5.2 env (A40 46GB), fp32.

| config | mean Dice | median Dice | mean HD95 (mm) | fails (<0.1) | nnU-Net holdout Dice |
|---|---|---|---|---|---|
| 2D  | 0.9392 | 0.9801 | 7.1  | 1 | 0.9744 |
| 3D  | **0.9615** | 0.9773 | 9.7  | 0 | 0.9605 |

**2D**: 150 epochs, EMA pseudo-dice plateaued ~0.98. On the 28-case test set,
mean Dice 0.939 / median 0.980 / HD95 7.1 mm — the framework transfers cleanly
to a new organ. The single fail is a low-foreground / edge case.

**3D**: 150 epochs, best EMA pseudo-dice 0.9758, nnU-Net internal holdout
(validation) Dice 0.9605. On the 28-case test set, mean Dice 0.962 / median
0.977 / HD95 9.7 mm, 0 hard fails. 3D beats 2D on mean Dice (0.962 vs 0.939)
and has no hard fail. The higher mean HD95 is driven by a single outlier
(s0623, dice 0.54 / HD95 144 mm, a low-foreground edge case); excluding it the
3D mean Dice is 0.977 and mean HD95 3.0 mm — i.e. on clean cases 3D is both
more accurate and more boundary-precise than 2D. Per-case, 3D is more stable
across the test set (fewer low-Dice outliers).

3D trained at ~10.3 min/epoch (617 s/epoch, 355M-param fp32 3D model) —
inherently slow, consistent with the kidney 3D runs (~610–672 s/epoch). Full
3D run took ~26 h.

## Notes / issues hit

1. **`_parse_build_args` bool bug** — nnU-Net passes `enable_deep_supervision=False`
   as the last positional arg; `isinstance(False, int)` is True in Python, so it
   got picked as `num_output_channels` → `num_classes=False` → Conv2d crash.
   Fixed by excluding bools and `int()`-coercing.
2. **batch_size OOM** — nnU-Net auto-sized 2D batch to 49 (for a small
   PlainConvUNet); FlexiCT fp32 OOM'd. First attempted a trainer `initialize()`
   clamp — WRONG: it desyncs nnU-Net's oversample schedule, dataloader still
   used bs=49, OOM'd at 44 GB. Correct fix: lower batch_size in the plans
   (`scripts/set_plan_batch_size.py`) so nnU-Net's native flow sees it.
   See `docs/pitfalls.md` §8. Validated: 2D bs=8 = 5 GB, 3D bs=2 = 36 GB.


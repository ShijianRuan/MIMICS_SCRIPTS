@echo off
cd /d E:\mimics_script_offline\external\flexict-finetune
python scripts/stream_uncertainty_rank.py --source "Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201" --remote wenwen_zhang@10.9.87.50 -p 11208 --remote-root "~/flexict-finetune" --dataset-id 907 --local-masks ./unc_masks --out ./unc_out --gpu-2d 0 --gpu-3d 2

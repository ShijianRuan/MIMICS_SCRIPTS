set -e

sleep 1m

echo "Hello World"

#nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres --verify_dataset_integrity -np 6

#CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres --verify_dataset_integrity -np 6
# nnUNetv2_plan_and_preprocess -d 203 -c 3d_fullres --verify_dataset_integrity -np 6
# nnUNetv2_plan_and_preprocess -d 204 -c 3d_fullres --verify_dataset_integrity -np 6
# nnUNetv2_plan_and_preprocess -d 205 -c 3d_fullres --verify_dataset_integrity -np 6
# nnUNetv2_plan_and_preprocess -d 206 -c 3d_fullres --verify_dataset_integrity -np 6


# cd /data1/segmentationForTrain/traindata/CTWholeBodyBone/nnUNet_raw/Dataset102_Rapid_Bone_3
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 102 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
# cd /data1/segmentationForTrain/traindata/MIv500_MR/nnUNet_raw/Dataset202_MR2_Chest
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 202 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
# cd /data1/segmentationForTrain/traindata/MIv500_MR/nnUNet_raw/Dataset203_MR3_Abdomen
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 203 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
# cd /data1/segmentationForTrain/traindata/MIv500_MR/nnUNet_raw/Dataset204_MR5_Vertebrae
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 204 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
# cd /data1/segmentationForTrain/traindata/MIv500_MR/nnUNet_raw/Dataset205_MR7_AbdomenBone
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 205 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0
# cd /data1/segmentationForTrain/traindata/MIv500_MR/nnUNet_raw/Dataset206_MR8_muscle
# nnUNetv2_predict -i imagesTs -o labelsTs_predicted -d 206 -c 3d_fullres -tr nnUNetTrainerNoMirroring --disable_tta -f 0


# python ../resources/evaluate.py labelsTs labelsTs_predicted class_map_part_organs

# nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerNoMirroring
# nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerNoMirroring
# nnUNetv2_train 103 3d_fullres 0 -tr nnUNetTrainerNoMirroring
# nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerNoMirroring

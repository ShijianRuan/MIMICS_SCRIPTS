#!/usr/bin/env bash
# Environment setup for FlexiCT few-shot finetuning.
# Source this BEFORE running nnUNetv2 commands:  source scripts/setup_env.sh
#
# Override any of these by exporting them first, e.g.
#   export nnUNet_results=/some/where  &&  source scripts/setup_env.sh

# Repo root = parent of this script's dir
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- nnU-Net paths (set to your local layout if different) ---
export nnUNet_raw="${nnUNet_raw:-$REPO_ROOT/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO_ROOT/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$REPO_ROOT/data/nnUNet_results}"
export nnUNet_compile="${nnUNet_compile:-0}"   # compile=0: FlexiCT dynamic shapes break torch.compile

# --- FlexiCT paths ---
export FLEXICT_EXT_DIR="${FLEXICT_EXT_DIR:-$REPO_ROOT/flexict}"
export FLEXICT2D_CKPT="${FLEXICT2D_CKPT:-$REPO_ROOT/weights/flexict_2d/model.safetensors}"
export FLEXICT3D_CKPT="${FLEXICT3D_CKPT:-$REPO_ROOT/weights/flexict_3d/model.safetensors}"

# --- training length / mirroring ---
export NUM_EPOCHS="${NUM_EPOCHS:-150}"
# For a single-side / asymmetric target, exclude the lateral axis from mirroring.
# Example: export MIRROR_DISABLE_AXES=1
# (unset by default = standard nnU-Net mirroring)

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
echo "[setup] nnUNet_raw=$nnUNet_raw"
echo "[setup] FLEXICT_EXT_DIR=$FLEXICT_EXT_DIR"
echo "[setup] NUM_EPOCHS=$NUM_EPOCHS  MIRROR_DISABLE_AXES=${MIRROR_DISABLE_AXES:-<none>}"

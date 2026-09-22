#!/usr/bin/env bash
set -euo pipefail

IMAGE="${MIMICS_AI_IMAGE:-mimics-ai-runtime:1.0}"
ROOT="${MIMICS_AI_ROOT:-$HOME/mimics-ai}"
IMAGE_ARCHIVE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Mimics-Script remote training setup"
echo "Root:  $ROOT"
echo "Image: $IMAGE"

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: Docker is not installed or is not available to this SSH user." >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: NVIDIA driver tools are not available." >&2
  exit 1
}

if [[ "$IMAGE_ARCHIVE" == "--build" ]]; then
  "$SCRIPT_DIR/build_image.sh"
elif [[ -n "$IMAGE_ARCHIVE" ]]; then
  [[ -f "$IMAGE_ARCHIVE" ]] || {
    echo "ERROR: Image archive does not exist: $IMAGE_ARCHIVE" >&2
    exit 1
  }
  docker load -i "$IMAGE_ARCHIVE"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: Docker image is not installed: $IMAGE" >&2
  echo "Load the supplied image archive or build remote/Dockerfile first." >&2
  exit 1
}

mkdir -p \
  "$ROOT/jobs" \
  "$ROOT/outputs" \
  "$ROOT/locks" \
  "$ROOT/cache" \
  "$ROOT/models/cache" \
  "$ROOT/models/nninteractive"

echo
echo "Expected base-model layout:"
echo "  $ROOT/models/nninteractive/nnInteractive_v1.0/"
echo

if [[ -d "$ROOT/models/dinov3" ]]; then
  echo "NOTE: $ROOT/models/dinov3 is no longer used and can be removed." >&2
fi
if [[ ! -d "$ROOT/models/nninteractive/nnInteractive_v1.0" ]] || \
  ! find "$ROOT/models/nninteractive/nnInteractive_v1.0" \
    -maxdepth 4 -type f \( -name '*.pth' -o -name '*.safetensors' \) \
    -print -quit | grep -q .; then
  echo "ERROR: Official nnInteractive weights are missing." >&2
  exit 1
fi

docker run --rm --gpus all --network none \
  -v "$ROOT/models:/models:ro" \
  "$IMAGE" \
  python /app/tools/remote_worker.py preflight --models-dir /models

echo
echo "Remote runtime is ready."
echo "The Docker image is installed once and reused by disposable job containers."
echo "Successful and cancelled containers are removed automatically."
echo "Verified training-data cache entries are retained for 30 days."
echo "Use this work folder in the client server profile: $ROOT"

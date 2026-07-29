#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MIMICS_AI_IMAGE:-mimics-ai-runtime:1.0}"
OUTPUT="${1:-}"

echo "Building unified Mimics-Script AI runtime: $IMAGE"
docker build \
  --file "$PROJECT_ROOT/remote/Dockerfile" \
  --tag "$IMAGE" \
  "$PROJECT_ROOT"

if [[ -n "$OUTPUT" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  docker save "$IMAGE" -o "$OUTPUT"
  echo "Saved image archive: $OUTPUT"
fi

echo "Image ready: $IMAGE"

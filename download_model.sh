#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="model"
MODEL_FILE="gemma-4-E2B-it-Q3_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_URL="https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q3_K_M.gguf?download=true"

mkdir -p "$MODEL_DIR"

if [ -s "$MODEL_PATH" ]; then
  echo "Model already exists: $MODEL_PATH"
  exit 0
fi

echo "Downloading $MODEL_FILE..."
curl -L --fail --retry 3 --continue-at - \
  -o "$MODEL_PATH.part" \
  "$MODEL_URL"

mv "$MODEL_PATH.part" "$MODEL_PATH"
echo "Model downloaded successfully: $MODEL_PATH"

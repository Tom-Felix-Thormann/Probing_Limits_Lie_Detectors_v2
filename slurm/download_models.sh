#!/bin/bash
set -e
cd "$(dirname "$0")"

hf download google/gemma-3-27b-it          --local-dir model_weights/gemma-3-27B-instruct
hf download Qwen/Qwen2.5-7B-Instruct       --local-dir model_weights/qwen-2.5-7B-instruct
hf download Qwen/Qwen2.5-72B-Instruct      --local-dir model_weights/qwen-2.5-72B-instruct
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir model_weights/mistral-7B-instruct \
    --exclude "consolidated.safetensors"
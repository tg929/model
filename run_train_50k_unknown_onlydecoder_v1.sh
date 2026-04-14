#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/ytg/model"
TRAIN_JSONL="${ROOT}/schneider50k/processed_only_decoder_unknown/train.jsonl"
VAL_JSONL="${ROOT}/schneider50k/processed_only_decoder_unknown/val.jsonl"
WEIGHT_PATH="${ROOT}/decoder/weights/SMILES-650M-3B-Epoch1.pt"
VOCAB_PATH="${ROOT}/decoder/vocabs/vocab.txt"

# Keep this run name stable for side-by-side comparisons.
OUTPUT_ROOT="${ROOT}/decoder_runs/uspto50k_unknown_onlydecoder_v1"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "missing train jsonl: ${TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${VAL_JSONL}" ]]; then
  echo "missing val jsonl: ${VAL_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHT_PATH}" ]]; then
  echo "missing base weight: ${WEIGHT_PATH}" >&2
  exit 1
fi

cd "${ROOT}"

conda run --no-capture-output -n retrogp \
  python decoder_runs/run_only_decoder_5epoch.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --weight-path "${WEIGHT_PATH}" \
  --model-size 650M \
  --vocab-path "${VOCAB_PATH}" \
  --num-epochs 5 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --learning-rate 3e-5 \
  --max-seq-len 256 \
  --save-every-steps 1000 \
  --val-checks-per-epoch 5 \
  --num-workers 2 \
  --device cuda

#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/ytg/model"
DATA_JSONL="${ROOT}/schneider50k/processed_only_decoder_unknown/test.jsonl"
CHECKPOINT="${ROOT}/decoder_runs/uspto50k_unknown_onlydecoder_v1/best.pt"
WEIGHT_PATH="${ROOT}/decoder/weights/SMILES-650M-3B-Epoch1.pt"
VOCAB_PATH="${ROOT}/decoder/vocabs/vocab.txt"
OUT_DIR="${ROOT}/decoder_test_results/uspto50k_unknown_onlydecoder_v1_best"

OUT_JSON="${OUT_DIR}/test_best_metrics.json"
OUT_PRED_JSONL="${OUT_DIR}/test_best_predictions.jsonl"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${DATA_JSONL}" ]]; then
  echo "missing test jsonl: ${DATA_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHT_PATH}" ]]; then
  echo "missing base weight: ${WEIGHT_PATH}" >&2
  exit 1
fi

cd "${ROOT}"

conda run --no-capture-output -n retrogp \
  python decoder/eval_retrosyn_only_decoder.py \
  --data-jsonl "${DATA_JSONL}" \
  --checkpoint "${CHECKPOINT}" \
  --weight-path "${WEIGHT_PATH}" \
  --model-size 650M \
  --vocab-path "${VOCAB_PATH}" \
  --output-json "${OUT_JSON}" \
  --predictions-jsonl "${OUT_PRED_JSONL}" \
  --beam-width 10 \
  --top-ks 1,3,5,10 \
  --max-new-tokens 256 \
  --length-penalty 0.0 \
  --save-every-samples 1000 \
  --device cuda

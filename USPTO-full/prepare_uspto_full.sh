#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m rxnutils.data.uspto.preparation_pipeline run \
  --folder . \
  --nbatches "${NBATCHES:-200}" \
  --max-workers "${MAX_WORKERS:-8}" \
  --max-num-splits "${MAX_NUM_SPLITS:-200}"

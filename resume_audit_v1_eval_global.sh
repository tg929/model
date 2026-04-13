#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASE_DIR="decoder_test_results/testall_epoch4_beamfix_audit_v1"
DATA_JSONL="USPTO-full/processed_only_decoder_audit_v1/test.jsonl"
CHECKPOINT="decoder_runs/only_decoder_650m_10epoch/best.pt"
WEIGHT_PATH="decoder/weights/SMILES-650M-3B-Epoch1.pt"
MODEL_SIZE="650M"
TOP_KS="1,3,5,10"
SAVE_EVERY=1000
LENGTH_PENALTY=0.0

PRED_MAIN="$BASE_DIR/testall_best_predictions.jsonl"
MET_MAIN="$BASE_DIR/testall_best_metrics.json"

LEGACY_PART_PRED="$BASE_DIR/testall_best_predictions_resume_part.jsonl"
LEGACY_PART_MET="$BASE_DIR/testall_best_metrics_resume_part.json"
LEGACY_PART_MET_GLOB="$BASE_DIR/testall_best_metrics_resume_part_up_to_"'*.json'
LEGACY_REMAIN_GLOB="$BASE_DIR/test_remaining_from_"'*.jsonl'
LEGACY_LOG="$BASE_DIR/resume_eval.log"
LEGACY_OLD_SCRIPT="$ROOT_DIR/resume_audit_v1_eval.sh"

PART_PRED="$BASE_DIR/testall_best_predictions_resume_global_part.jsonl"
PART_MET="$BASE_DIR/testall_best_metrics_resume_global_part.json"
PART_MET_PREFIX="${PART_MET%.json}"
RUN_LOG="$BASE_DIR/resume_global_eval.log"
BASE_STATS_JSON="$BASE_DIR/.resume_global_base_stats.json"

count_valid_jsonl() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
n = 0
with open(path, "r", encoding="utf-8", errors="replace") as fin:
    for i, line in enumerate(fin, 1):
        s = line.strip()
        if not s or set(s) == {"\x00"}:
            break
        try:
            json.loads(s)
        except Exception:
            break
        n = i
print(n)
PY
}

trim_file_to_valid_jsonl() {
  local path="$1"
  local valid="$2"
  local total
  total="$(wc -l < "$path" | tr -d ' ')"
  if [[ "$total" != "$valid" ]]; then
    local backup="${path}.trim_backup.$(date +%Y%m%d_%H%M%S)"
    cp "$path" "$backup"
    head -n "$valid" "$backup" > "$path"
    echo "trimmed invalid tail: $path (backup: $backup)"
  fi
}

verify_legacy_alignment() {
  local start_line="$1"
  local full_path="$2"
  local legacy_pred="$3"
  python - "$start_line" "$full_path" "$legacy_pred" <<'PY'
import json
import sys

start = int(sys.argv[1])
full = sys.argv[2]
legacy = sys.argv[3]

checked = 0
with open(full, "r", encoding="utf-8") as f_full, open(legacy, "r", encoding="utf-8") as f_legacy:
    for _ in range(start - 1):
        next(f_full)
    for i, (line_full, line_legacy) in enumerate(zip(f_full, f_legacy), 1):
        r_full = json.loads(line_full)
        r_legacy = json.loads(line_legacy)
        checked = i
        if r_full.get("product") != r_legacy.get("product") or r_full.get("target_text") != r_legacy.get("target_text"):
            raise RuntimeError(
                f"legacy alignment mismatch at relative={i}, global={start+i-1}"
            )
print(checked)
PY
}

write_prefix_metrics_and_base_stats() {
  local pred_path="$1"
  local limit="$2"
  local output_json="$3"
  local save_every="$4"
  local checkpoint="$5"
  local data_jsonl="$6"
  local base_stats_path="$7"
  local length_penalty="$8"
  local completed="$9"
  python - "$pred_path" "$limit" "$output_json" "$save_every" "$checkpoint" "$data_jsonl" "$base_stats_path" "$length_penalty" "$completed" <<'PY'
import json
import sys
from pathlib import Path

pred_path = Path(sys.argv[1])
limit = int(sys.argv[2])
output_json = Path(sys.argv[3])
save_every = int(sys.argv[4])
checkpoint = sys.argv[5]
data_jsonl = sys.argv[6]
base_stats_path = Path(sys.argv[7])
length_penalty = float(sys.argv[8])
completed = sys.argv[9].lower() == "true"
top_ks = [1, 3, 5, 10]

def build_metrics(total, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, done):
    return {
        "num_samples": total,
        "beam_width": 10,
        "top_ks": top_ks,
        "length_penalty": length_penalty,
        "save_every_samples": save_every,
        "completed": done,
        "topk_exact_match": {str(k): (exact_hits[k] / total if total else 0.0) for k in top_ks},
        "topk_canonical_match": {str(k): (canonical_hits[k] / total if total else 0.0) for k in top_ks},
        "topk_maxfrag_match": {str(k): (maxfrag_hits[k] / total if total else 0.0) for k in top_ks},
        "top1_invalid_smiles_rate": (invalid_top1 / total if total else 0.0),
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
    }

def write_metrics(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

def snapshot_path(path, total):
    return path.with_name(f"{path.stem}_up_to_{total}{path.suffix}")

exact_hits = {k: 0 for k in top_ks}
canonical_hits = {k: 0 for k in top_ks}
maxfrag_hits = {k: 0 for k in top_ks}
invalid_top1 = 0
total = 0

with pred_path.open("r", encoding="utf-8") as fin:
    for i, line in enumerate(fin, 1):
        if i > limit:
            break
        s = line.strip()
        if not s:
            continue
        row = json.loads(s)
        preds = row["predictions"]
        can_preds = row["canonical_predictions"]
        max_preds = row["maxfrag_predictions"]
        target = row["target_text"]
        can_target = row["canonical_target"]
        max_target = row["maxfrag_target"]

        total += 1
        if can_preds and can_preds[0] is None:
            invalid_top1 += 1

        for k in top_ks:
            if target in preds[:k]:
                exact_hits[k] += 1
            if can_target is not None and can_target in can_preds[:k]:
                canonical_hits[k] += 1
            if max_target is not None and max_target in max_preds[:k]:
                maxfrag_hits[k] += 1

        if total % save_every == 0:
            write_metrics(snapshot_path(output_json, total), build_metrics(total, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, False))

final_metrics = build_metrics(total, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, completed)
write_metrics(output_json, final_metrics)
if total and (total % save_every != 0):
    write_metrics(snapshot_path(output_json, total), build_metrics(total, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, False))

base_stats = {
    "num_samples": total,
    "exact_hits": exact_hits,
    "canonical_hits": canonical_hits,
    "maxfrag_hits": maxfrag_hits,
    "invalid_top1": invalid_top1,
}
base_stats_path.write_text(json.dumps(base_stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({"event": "base_metrics_ready", "num_samples": total}, ensure_ascii=True))
PY
}

write_global_snapshot_from_local_n() {
  local local_n="$1"
  python - "$BASE_STATS_JSON" "$PART_PRED" "$local_n" "$MET_MAIN" "$SAVE_EVERY" "$CHECKPOINT" "$DATA_JSONL" "$LENGTH_PENALTY" <<'PY'
import json
import sys
from pathlib import Path

base_stats_path = Path(sys.argv[1])
part_pred_path = Path(sys.argv[2])
local_n = int(sys.argv[3])
main_metrics_path = Path(sys.argv[4])
save_every = int(sys.argv[5])
checkpoint = sys.argv[6]
data_jsonl = sys.argv[7]
length_penalty = float(sys.argv[8])
top_ks = [1, 3, 5, 10]

base = json.loads(base_stats_path.read_text(encoding="utf-8"))
base_total = int(base["num_samples"])
base_exact = {int(k): int(v) for k, v in base["exact_hits"].items()}
base_canon = {int(k): int(v) for k, v in base["canonical_hits"].items()}
base_maxfrag = {int(k): int(v) for k, v in base["maxfrag_hits"].items()}
base_invalid = int(base["invalid_top1"])

part_exact = {k: 0 for k in top_ks}
part_canon = {k: 0 for k in top_ks}
part_maxfrag = {k: 0 for k in top_ks}
part_invalid = 0
part_total = 0

with part_pred_path.open("r", encoding="utf-8") as fin:
    for i, line in enumerate(fin, 1):
        if i > local_n:
            break
        row = json.loads(line)
        preds = row["predictions"]
        can_preds = row["canonical_predictions"]
        max_preds = row["maxfrag_predictions"]
        target = row["target_text"]
        can_target = row["canonical_target"]
        max_target = row["maxfrag_target"]
        part_total += 1
        if can_preds and can_preds[0] is None:
            part_invalid += 1
        for k in top_ks:
            if target in preds[:k]:
                part_exact[k] += 1
            if can_target is not None and can_target in can_preds[:k]:
                part_canon[k] += 1
            if max_target is not None and max_target in max_preds[:k]:
                part_maxfrag[k] += 1

if part_total < local_n:
    raise RuntimeError(f"part predictions has only {part_total} rows, less than requested local_n={local_n}")

total = base_total + part_total
exact = {k: base_exact[k] + part_exact[k] for k in top_ks}
canon = {k: base_canon[k] + part_canon[k] for k in top_ks}
maxfrag = {k: base_maxfrag[k] + part_maxfrag[k] for k in top_ks}
invalid = base_invalid + part_invalid

metrics = {
    "num_samples": total,
    "beam_width": 10,
    "top_ks": top_ks,
    "length_penalty": length_penalty,
    "save_every_samples": save_every,
    "completed": False,
    "topk_exact_match": {str(k): exact[k] / total for k in top_ks},
    "topk_canonical_match": {str(k): canon[k] / total for k in top_ks},
    "topk_maxfrag_match": {str(k): maxfrag[k] / total for k in top_ks},
    "top1_invalid_smiles_rate": invalid / total,
    "checkpoint": checkpoint,
    "data_jsonl": data_jsonl,
}

main_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
snap = main_metrics_path.with_name(f"{main_metrics_path.stem}_up_to_{total}{main_metrics_path.suffix}")
snap.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({"event": "global_progress_save", "local_n": local_n, "global_n": total}, ensure_ascii=True))
PY
}

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "missing data jsonl: $DATA_JSONL" >&2
  exit 1
fi
if [[ ! -f "$PRED_MAIN" ]]; then
  echo "missing main prediction file: $PRED_MAIN" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$WEIGHT_PATH" ]]; then
  echo "missing weight path: $WEIGHT_PATH" >&2
  exit 1
fi

TOTAL_ROWS="$(wc -l < "$DATA_JSONL" | tr -d ' ')"
MAIN_DONE="$(count_valid_jsonl "$PRED_MAIN")"
trim_file_to_valid_jsonl "$PRED_MAIN" "$MAIN_DONE"

if [[ -f "$LEGACY_PART_PRED" ]]; then
  LEGACY_ROWS="$(count_valid_jsonl "$LEGACY_PART_PRED")"
  if (( LEGACY_ROWS > 0 )); then
    START_LINE=$((MAIN_DONE + 1))
    END_LINE=$((MAIN_DONE + LEGACY_ROWS))
    if (( END_LINE > TOTAL_ROWS )); then
      echo "legacy resume rows overflow: main_done=$MAIN_DONE legacy_rows=$LEGACY_ROWS total=$TOTAL_ROWS" >&2
      exit 1
    fi
    VERIFIED="$(verify_legacy_alignment "$START_LINE" "$DATA_JSONL" "$LEGACY_PART_PRED")"
    if (( VERIFIED != LEGACY_ROWS )); then
      echo "legacy verify rows mismatch: verified=$VERIFIED legacy_rows=$LEGACY_ROWS" >&2
      exit 1
    fi
    MAIN_BACKUP="${PRED_MAIN}.before_legacy_merge.$(date +%Y%m%d_%H%M%S)"
    cp "$PRED_MAIN" "$MAIN_BACKUP"
    cat "$LEGACY_PART_PRED" >> "$PRED_MAIN"
    MAIN_DONE=$((MAIN_DONE + LEGACY_ROWS))
    echo "merged legacy resume part: +$LEGACY_ROWS rows -> main_done=$MAIN_DONE (backup: $MAIN_BACKUP)"
  fi
fi

write_prefix_metrics_and_base_stats \
  "$PRED_MAIN" \
  "$MAIN_DONE" \
  "$MET_MAIN" \
  "$SAVE_EVERY" \
  "$CHECKPOINT" \
  "$DATA_JSONL" \
  "$BASE_STATS_JSON" \
  "$LENGTH_PENALTY" \
  "false"

echo "normalized cumulative metrics ready: num_samples=$MAIN_DONE"

if (( MAIN_DONE >= TOTAL_ROWS )); then
  write_prefix_metrics_and_base_stats \
    "$PRED_MAIN" \
    "$MAIN_DONE" \
    "$MET_MAIN" \
    "$SAVE_EVERY" \
    "$CHECKPOINT" \
    "$DATA_JSONL" \
    "$BASE_STATS_JSON" \
    "$LENGTH_PENALTY" \
    "true"
  rm -f "$BASE_STATS_JSON"
  echo "already completed, no remaining samples."
  exit 0
fi

REMAIN_START=$((MAIN_DONE + 1))
REMAIN_ROWS=$((TOTAL_ROWS - MAIN_DONE))
REMAIN_JSONL="$BASE_DIR/test_remaining_from_${REMAIN_START}.jsonl"
tail -n +"$REMAIN_START" "$DATA_JSONL" > "$REMAIN_JSONL"
echo "remaining data: $REMAIN_JSONL ($REMAIN_ROWS rows)"

rm -f "$PART_PRED" "$PART_MET" "$RUN_LOG"
rm -f "${PART_MET_PREFIX}"_up_to_*.json 2>/dev/null || true

conda run -n retrogp python decoder/eval_retrosyn_only_decoder.py \
  --data-jsonl "$REMAIN_JSONL" \
  --checkpoint "$CHECKPOINT" \
  --weight-path "$WEIGHT_PATH" \
  --model-size "$MODEL_SIZE" \
  --output-json "$PART_MET" \
  --predictions-jsonl "$PART_PRED" \
  --beam-width 10 \
  --top-ks "$TOP_KS" \
  --max-new-tokens 256 \
  --length-penalty "$LENGTH_PENALTY" \
  --save-every-samples "$SAVE_EVERY" \
  --device cuda \
  > "$RUN_LOG" 2>&1 &

EVAL_PID=$!
echo "eval started: pid=$EVAL_PID, log=$RUN_LOG"

LAST_SYNC_N=0
while kill -0 "$EVAL_PID" 2>/dev/null; do
  for metric_file in $(ls "${PART_MET_PREFIX}"_up_to_*.json 2>/dev/null | sort -V); do
    local_n="${metric_file##*_up_to_}"
    local_n="${local_n%.json}"
    if [[ -z "$local_n" ]]; then
      continue
    fi
    if (( local_n <= LAST_SYNC_N )); then
      continue
    fi
    if write_global_snapshot_from_local_n "$local_n"; then
      LAST_SYNC_N="$local_n"
    fi
  done
  sleep 30
done

wait "$EVAL_PID"

for metric_file in $(ls "${PART_MET_PREFIX}"_up_to_*.json 2>/dev/null | sort -V); do
  local_n="${metric_file##*_up_to_}"
  local_n="${local_n%.json}"
  if [[ -z "$local_n" ]]; then
    continue
  fi
  if (( local_n <= LAST_SYNC_N )); then
    continue
  fi
  if write_global_snapshot_from_local_n "$local_n"; then
    LAST_SYNC_N="$local_n"
  fi
done

if [[ ! -f "$PART_PRED" ]]; then
  echo "missing part predictions after eval: $PART_PRED" >&2
  exit 1
fi

PART_DONE="$(count_valid_jsonl "$PART_PRED")"
trim_file_to_valid_jsonl "$PART_PRED" "$PART_DONE"
if (( PART_DONE != REMAIN_ROWS )); then
  echo "incomplete part eval: expected=$REMAIN_ROWS got=$PART_DONE" >&2
  exit 1
fi

FINAL_BACKUP="${PRED_MAIN}.before_final_merge.$(date +%Y%m%d_%H%M%S)"
cp "$PRED_MAIN" "$FINAL_BACKUP"
cat "$PART_PRED" >> "$PRED_MAIN"
FINAL_DONE="$(count_valid_jsonl "$PRED_MAIN")"
if (( FINAL_DONE != TOTAL_ROWS )); then
  echo "final merged predictions mismatch: expected=$TOTAL_ROWS got=$FINAL_DONE" >&2
  exit 1
fi
echo "final merge complete: total=$FINAL_DONE (backup: $FINAL_BACKUP)"

write_prefix_metrics_and_base_stats \
  "$PRED_MAIN" \
  "$FINAL_DONE" \
  "$MET_MAIN" \
  "$SAVE_EVERY" \
  "$CHECKPOINT" \
  "$DATA_JSONL" \
  "$BASE_STATS_JSON" \
  "$LENGTH_PENALTY" \
  "true"

rm -f "$BASE_STATS_JSON"
echo "final metrics completed: $MET_MAIN"

echo "cleanup legacy temporary artifacts..."
rm -f "$LEGACY_PART_PRED" "$LEGACY_PART_MET" "$LEGACY_LOG"
rm -f $LEGACY_PART_MET_GLOB 2>/dev/null || true
rm -f $LEGACY_REMAIN_GLOB 2>/dev/null || true
rm -f "$PART_PRED" "$PART_MET"
rm -f "${PART_MET_PREFIX}"_up_to_*.json 2>/dev/null || true

if [[ -f "$LEGACY_OLD_SCRIPT" ]]; then
  rm -f "$LEGACY_OLD_SCRIPT"
fi

echo "resume workflow finished."

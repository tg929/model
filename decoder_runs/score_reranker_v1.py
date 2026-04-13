#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
DECODER_DIR = REPO_ROOT / "decoder"
if str(DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_DIR))

from loadmodel_example import DEFAULT_VOCAB_PATH, load_pretrained_model  # noqa: E402


DEFAULT_INPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_input.jsonl"
DEFAULT_CHECKPOINT = REPO_ROOT / "decoder_runs" / "only_decoder_650m_10epoch" / "best.pt"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "decoder" / "weights" / "SMILES-650M-3B-Epoch1.pt"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_scored.jsonl"
DEFAULT_METRICS_JSON = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_metrics.json"

SCORE_NAMES = ("mean_target_eos", "mean_target", "sum_target_eos")
TOP_KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-force beam candidates and score reranker v1 outputs.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--model-size", type=str, default="650M")
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--save-every-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Ignore any existing scored output and start from scratch.")
    return parser.parse_args()


def load_checkpoint_model(args: argparse.Namespace):
    model, tokenizer, device = load_pretrained_model(
        weight_path=args.weight_path,
        model_size=args.model_size,
        vocab_path=args.vocab_path,
        device=args.device,
    )
    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model, tokenizer, device


def build_scoring_batch(source_ids: list[int], candidates: list[str], tokenizer, device: str):
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    input_rows = []
    labels_rows = []
    target_mask_rows = []
    target_eos_mask_rows = []
    target_lens = []

    for text in candidates:
        target_ids = tokenizer.encode(text, add_special_tokens=False)
        seq = [bos_id] + source_ids + target_ids + [eos_id]
        input_ids = seq[:-1]
        labels = seq[1:]
        source_len = len(source_ids)
        labels[:source_len] = [pad_id] * source_len

        target_mask = [False] * len(input_ids)
        target_eos_mask = [False] * len(input_ids)
        for pos in range(source_len, source_len + len(target_ids)):
            target_mask[pos] = True
        for pos in range(source_len, source_len + len(target_ids) + 1):
            target_eos_mask[pos] = True

        input_rows.append(input_ids)
        labels_rows.append(labels)
        target_mask_rows.append(target_mask)
        target_eos_mask_rows.append(target_eos_mask)
        target_lens.append(len(target_ids))

    max_len = max(len(row) for row in input_rows)

    def pad_int_rows(rows: list[list[int]], value: int) -> torch.Tensor:
        padded = [row + [value] * (max_len - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.long, device=device)

    def pad_bool_rows(rows: list[list[bool]]) -> torch.Tensor:
        padded = [row + [False] * (max_len - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.bool, device=device)

    return {
        "input_ids": pad_int_rows(input_rows, pad_id),
        "labels": pad_int_rows(labels_rows, pad_id),
        "target_mask": pad_bool_rows(target_mask_rows),
        "target_eos_mask": pad_bool_rows(target_eos_mask_rows),
        "target_lens": target_lens,
    }


@torch.inference_mode()
def score_candidates(model, tokenizer, source_text: str, candidates: list[str], device: str) -> dict[str, list[float | None]]:
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    batch = build_scoring_batch(source_ids, candidates, tokenizer, device)
    logits, _, _ = model(batch["input_ids"], tokenizer)
    log_probs = F.log_softmax(logits, dim=-1)

    labels = batch["labels"]
    gather_labels = labels.masked_fill(labels == tokenizer.pad_token_id, 0)
    gathered = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    sum_target = (gathered * batch["target_mask"]).sum(dim=1)
    sum_target_eos = (gathered * batch["target_eos_mask"]).sum(dim=1)
    count_target = batch["target_mask"].sum(dim=1)
    count_target_eos = batch["target_eos_mask"].sum(dim=1)

    mean_target = []
    for score, count in zip(sum_target.tolist(), count_target.tolist()):
        mean_target.append(None if count == 0 else score / count)

    return {
        "mean_target_eos": [score / count for score, count in zip(sum_target_eos.tolist(), count_target_eos.tolist())],
        "mean_target": mean_target,
        "sum_target_eos": sum_target_eos.tolist(),
    }


def score_candidates_safe(model, tokenizer, source_text: str, candidates: list[str], device: str) -> tuple[dict[str, list[float | None]], list[bool]]:
    try:
        return score_candidates(model, tokenizer, source_text, candidates, device), [False] * len(candidates)
    except Exception:
        all_scores = {name: [] for name in SCORE_NAMES}
        failed = []
        for candidate in candidates:
            try:
                one = score_candidates(model, tokenizer, source_text, [candidate], device)
                for name in SCORE_NAMES:
                    all_scores[name].append(one[name][0])
                failed.append(False)
            except Exception:
                for name in SCORE_NAMES:
                    all_scores[name].append(None)
                failed.append(True)
        return all_scores, failed


def sort_indices(scores: list[float | None]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: (-math.inf if scores[idx] is None else scores[idx], -idx), reverse=True)


def hits_for_order(order: list[int], target_text: str, canonical_target: str | None, maxfrag_target: str | None, candidates: list[dict]) -> dict[str, dict[int, int]]:
    exact = {k: 0 for k in TOP_KS}
    canonical = {k: 0 for k in TOP_KS}
    maxfrag = {k: 0 for k in TOP_KS}
    for k in TOP_KS:
        top = order[:k]
        texts = [candidates[idx]["text"] for idx in top]
        canonical_texts = [candidates[idx]["canonical_text"] for idx in top]
        maxfrag_texts = [candidates[idx]["maxfrag_text"] for idx in top]
        exact[k] = int(target_text in texts)
        canonical[k] = int(canonical_target is not None and canonical_target in canonical_texts)
        maxfrag[k] = int(maxfrag_target is not None and maxfrag_target in maxfrag_texts)
    return {"exact": exact, "canonical": canonical, "maxfrag": maxfrag}


def empty_metric_bucket() -> dict[str, dict[int, int] | int]:
    return {
        "exact": {k: 0 for k in TOP_KS},
        "canonical": {k: 0 for k in TOP_KS},
        "maxfrag": {k: 0 for k in TOP_KS},
        "invalid_top1": 0,
    }


def add_hits(bucket: dict, hits: dict, invalid_top1: int) -> None:
    for name in ("exact", "canonical", "maxfrag"):
        for k in TOP_KS:
            bucket[name][k] += hits[name][k]
    bucket["invalid_top1"] += invalid_top1


def bucket_to_rates(bucket: dict, total: int) -> dict:
    return {
        "topk_exact_match": {str(k): bucket["exact"][k] / total if total else 0.0 for k in TOP_KS},
        "topk_canonical_match": {str(k): bucket["canonical"][k] / total if total else 0.0 for k in TOP_KS},
        "topk_maxfrag_match": {str(k): bucket["maxfrag"][k] / total if total else 0.0 for k in TOP_KS},
        "top1_invalid_smiles_rate": bucket["invalid_top1"] / total if total else 0.0,
    }


def metrics_snapshot_path(path: Path, total: int) -> Path:
    return path.with_name(f"{path.stem}_up_to_{total}{path.suffix}")


def build_metrics(args: argparse.Namespace, summary: dict, total: int, scoring_failed_samples: int, scoring_failed_candidates: int, completed: bool) -> dict:
    return {
        "input_jsonl": str(args.input_jsonl),
        "checkpoint": str(args.checkpoint),
        "weight_path": str(args.weight_path),
        "model_size": args.model_size,
        "main_score": "mean_target_eos",
        "score_names": list(SCORE_NAMES),
        "save_every_samples": args.save_every_samples,
        "num_samples": total,
        "completed": completed,
        "scoring_failed_samples": scoring_failed_samples,
        "scoring_failed_candidates": scoring_failed_candidates,
        "before": bucket_to_rates(summary["before"], total),
        **{name: bucket_to_rates(summary[name], total) for name in SCORE_NAMES},
    }


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=True)
        fout.write("\n")


def rebuild_progress_from_output(path: Path) -> tuple[dict, int, int, int, bool]:
    summary = {
        "before": empty_metric_bucket(),
        **{name: empty_metric_bucket() for name in SCORE_NAMES},
    }
    total = 0
    scoring_failed_samples = 0
    scoring_failed_candidates = 0
    truncated = False

    with path.open("r+", encoding="utf-8") as fin:
        while True:
            offset = fin.tell()
            line = fin.readline()
            if not line:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                fin.truncate(offset)
                truncated = True
                break

            candidates = row["candidates"]
            failed_mask = row.get("scoring_failed_mask", [])
            if any(failed_mask):
                scoring_failed_samples += 1
                scoring_failed_candidates += sum(bool(flag) for flag in failed_mask)

            before_order = list(range(len(candidates)))
            before_hits = hits_for_order(before_order, row["target_text"], row["canonical_target"], row["maxfrag_target"], candidates)
            add_hits(summary["before"], before_hits, int(bool(candidates) and candidates[0]["canonical_text"] is None))

            reranked_indices = row["reranked_candidate_indices"]
            for name in SCORE_NAMES:
                order = reranked_indices[name]
                hits = hits_for_order(order, row["target_text"], row["canonical_target"], row["maxfrag_target"], candidates)
                invalid_top1 = int(bool(order) and candidates[order[0]]["canonical_text"] is None)
                add_hits(summary[name], hits, invalid_top1)

            total += 1

    return summary, total, scoring_failed_samples, scoring_failed_candidates, truncated


def acquire_output_lock(output_jsonl: Path):
    lock_path = output_jsonl.with_name(f"{output_jsonl.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(f"another reranker writer is already active for {output_jsonl}") from exc
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file, lock_path


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")
    if args.save_every_samples < 1:
        raise ValueError("--save-every-samples must be at least 1.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    lock_file, lock_path = acquire_output_lock(args.output_jsonl)
    summary = {
        "before": empty_metric_bucket(),
        **{name: empty_metric_bucket() for name in SCORE_NAMES},
    }
    total = 0
    scoring_failed_samples = 0
    scoring_failed_candidates = 0
    truncated_existing_output = False

    try:
        if args.output_jsonl.exists() and not args.overwrite:
            (
                summary,
                total,
                scoring_failed_samples,
                scoring_failed_candidates,
                truncated_existing_output,
            ) = rebuild_progress_from_output(args.output_jsonl)
            print(json.dumps({"event": "resume_scan", "num_samples": total, "output_jsonl": str(args.output_jsonl), "truncated_partial_line": truncated_existing_output}, ensure_ascii=True))

        if args.max_samples is not None and total >= args.max_samples:
            metrics = build_metrics(args, summary, total, scoring_failed_samples, scoring_failed_candidates, completed=(total == args.max_samples))
            write_metrics(args.metrics_json, metrics)
            write_metrics(metrics_snapshot_path(args.metrics_json, total), metrics)
            print(json.dumps({"event": "already_complete_for_requested_max_samples", "num_samples": total}, ensure_ascii=True))
            print(json.dumps(metrics, indent=2, ensure_ascii=True))
            return

        model, tokenizer, device = load_checkpoint_model(args)
        output_mode = "a" if total else "w"

        with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_jsonl.open(output_mode, encoding="utf-8", buffering=1) as fout:
            for line_idx, line in enumerate(fin):
                if line_idx < total:
                    continue
                if args.max_samples is not None and line_idx >= args.max_samples:
                    break

                row = json.loads(line)
                candidates = row["candidates"]
                candidate_texts = [item["text"] for item in candidates]
                scores, failed_mask = score_candidates_safe(model, tokenizer, row["source_text"], candidate_texts, device)
                failure_count = sum(failed_mask)
                if failure_count:
                    scoring_failed_samples += 1
                    scoring_failed_candidates += failure_count

                before_order = list(range(len(candidates)))
                before_hits = hits_for_order(before_order, row["target_text"], row["canonical_target"], row["maxfrag_target"], candidates)
                add_hits(summary["before"], before_hits, int(candidates[0]["canonical_text"] is None))

                reranked_indices = {}
                top1_after = {}
                for name in SCORE_NAMES:
                    order = sort_indices(scores[name])
                    reranked_indices[name] = order
                    top1_after[name] = candidates[order[0]]["text"] if order else None
                    hits = hits_for_order(order, row["target_text"], row["canonical_target"], row["maxfrag_target"], candidates)
                    invalid_top1 = int(order and candidates[order[0]]["canonical_text"] is None)
                    add_hits(summary[name], hits, invalid_top1)

                out_row = {
                    **row,
                    "scores": scores,
                    "scoring_failed_mask": failed_mask,
                    "reranked_candidate_indices": reranked_indices,
                    "top1_before": candidates[0]["text"] if candidates else None,
                    "top1_after_mean_target_eos": top1_after["mean_target_eos"],
                    "top1_after_mean_target": top1_after["mean_target"],
                    "top1_after_sum_target_eos": top1_after["sum_target_eos"],
                }
                fout.write(json.dumps(out_row, ensure_ascii=True) + "\n")
                total += 1

                if total % args.save_every_samples == 0:
                    fout.flush()
                    os.fsync(fout.fileno())
                    metrics = build_metrics(args, summary, total, scoring_failed_samples, scoring_failed_candidates, completed=False)
                    write_metrics(args.metrics_json, metrics)
                    write_metrics(metrics_snapshot_path(args.metrics_json, total), metrics)
                    print(json.dumps({"event": "progress_save", "num_samples": total, "metrics_json": str(args.metrics_json)}, ensure_ascii=True))

            fout.flush()
            os.fsync(fout.fileno())

        metrics = build_metrics(args, summary, total, scoring_failed_samples, scoring_failed_candidates, completed=True)
        write_metrics(args.metrics_json, metrics)
        if total % args.save_every_samples != 0:
            write_metrics(metrics_snapshot_path(args.metrics_json, total), metrics)

        print(json.dumps(metrics, indent=2, ensure_ascii=True))
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            if lock_path.exists():
                lock_path.unlink()


if __name__ == "__main__":
    main()

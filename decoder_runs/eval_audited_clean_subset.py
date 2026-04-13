#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_CSV = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_audit.csv"
DEFAULT_TEST_JSONL = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "test.jsonl"
DEFAULT_PREDICTIONS_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "testall_best_predictions.jsonl"
DEFAULT_RERANKER_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_scored.jsonl"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_effective_subset.jsonl"
DEFAULT_METRICS_JSON = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_clean_subset_metrics.json"

TOP_KS = (1, 3, 5, 10)
FOCUS_SMILES = {
    "THF": "C1CCOC1",
    "Et3N": "CCN(CC)CC",
}
ALLOWED_JUDGMENTS = {"", "true_contributor", "non_contributing_process_molecule", "ambiguous"}
ALLOWED_ACTIONS = {"", "keep_as_is", "remove_focus_molecule", "exclude_row", "unclear"}
ALLOWED_ROOT_CAUSES = {"", "mapping_leak", "role_merge_issue", "rule_too_permissive", "model_error", "unclear"}
MODEL_NAMES = ("before", "mean_target_eos", "mean_target", "sum_target_eos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an audited THF / Et3N clean subset against baseline and reranker outputs.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--predictions-jsonl", type=Path, default=DEFAULT_PREDICTIONS_JSONL)
    parser.add_argument("--reranker-jsonl", type=Path, default=DEFAULT_RERANKER_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    return parser.parse_args()


def split_components(text: str | None) -> list[str]:
    if not text:
        return []
    return [part for part in text.split(".") if part]


def canonicalize_reactants(text: str) -> str | None:
    parts = split_components(text)
    if not parts:
        return None

    canonical_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canonical_parts.append(Chem.MolToSmiles(mol, canonical=True))

    return ".".join(sorted(canonical_parts))


def largest_fragment(text: str) -> str | None:
    parts = split_components(text)
    if not parts:
        return None

    ranked = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        ranked.append((mol.GetNumAtoms(), Chem.MolToSmiles(mol, canonical=True)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def empty_metric_bucket() -> dict[str, dict[int, int] | int]:
    return {
        "exact": {k: 0 for k in TOP_KS},
        "canonical": {k: 0 for k in TOP_KS},
        "maxfrag": {k: 0 for k in TOP_KS},
        "invalid_top1": 0,
    }


def finalize_metric_bucket(bucket: dict, total: int) -> dict:
    return {
        "num_rows": total,
        "topk_exact_match": {str(k): bucket["exact"][k] / total if total else 0.0 for k in TOP_KS},
        "topk_canonical_match": {str(k): bucket["canonical"][k] / total if total else 0.0 for k in TOP_KS},
        "topk_maxfrag_match": {str(k): bucket["maxfrag"][k] / total if total else 0.0 for k in TOP_KS},
        "top1_invalid_smiles_rate": bucket["invalid_top1"] / total if total else 0.0,
    }


def update_metric_bucket(
    bucket: dict,
    target_text: str,
    canonical_target: str | None,
    maxfrag_target: str | None,
    predictions: list[str],
    canonical_predictions: list[str | None],
    maxfrag_predictions: list[str | None],
) -> None:
    if canonical_predictions and canonical_predictions[0] is None:
        bucket["invalid_top1"] += 1

    for k in TOP_KS:
        top_preds = predictions[:k]
        if target_text in top_preds:
            bucket["exact"][k] += 1
        if canonical_target is not None and canonical_target in canonical_predictions[:k]:
            bucket["canonical"][k] += 1
        if maxfrag_target is not None and maxfrag_target in maxfrag_predictions[:k]:
            bucket["maxfrag"][k] += 1


def find_exact_rank(target_text: str, predictions: list[str]) -> int | None:
    for rank, text in enumerate(predictions, start=1):
        if text == target_text:
            return rank
    return None


def validate_audit_row(row: dict[str, str]) -> None:
    if row["focus_molecule_judgment"] not in ALLOWED_JUDGMENTS:
        raise ValueError(f"invalid focus_molecule_judgment for sample_idx={row['sample_idx']}: {row['focus_molecule_judgment']}")
    if row["target_action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid target_action for sample_idx={row['sample_idx']}: {row['target_action']}")
    if row["root_cause_hypothesis"] not in ALLOWED_ROOT_CAUSES:
        raise ValueError(f"invalid root_cause_hypothesis for sample_idx={row['sample_idx']}: {row['root_cause_hypothesis']}")
    if row["focus_molecule"] not in FOCUS_SMILES:
        raise ValueError(f"unsupported focus_molecule for sample_idx={row['sample_idx']}: {row['focus_molecule']}")


def load_audit_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            validate_audit_row(row)
            rows.append(row)
    return rows


def load_selected_jsonl_rows(path: Path, sample_indices: set[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if line_idx in sample_indices:
                rows[line_idx] = json.loads(line)
    return rows


def load_selected_reranker_rows(path: Path, sample_indices: set[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            sample_idx = int(row["sample_idx"])
            if sample_idx in sample_indices:
                rows[sample_idx] = row
    return rows


def apply_target_action(target_text: str, focus_smiles: str, action: str) -> str | None:
    if action == "keep_as_is":
        return target_text
    if action == "exclude_row" or action == "unclear" or action == "":
        return None
    if action != "remove_focus_molecule":
        raise ValueError(f"unsupported action: {action}")

    kept = [component for component in split_components(target_text) if component != focus_smiles]
    if len(kept) == len(split_components(target_text)):
        raise ValueError(f"remove_focus_molecule requested but focus molecule not found in target: {target_text}")
    if not kept:
        raise ValueError(f"remove_focus_molecule would leave an empty target: {target_text}")
    return ".".join(kept)


def baseline_predictions(prediction_row: dict) -> tuple[list[str], list[str | None], list[str | None]]:
    return prediction_row["predictions"], prediction_row["canonical_predictions"], prediction_row["maxfrag_predictions"]


def reranker_predictions(reranker_row: dict, score_name: str) -> tuple[list[str], list[str | None], list[str | None]]:
    if score_name == "before":
        ordered_candidates = reranker_row["candidates"]
    else:
        order = reranker_row["reranked_candidate_indices"][score_name]
        ordered_candidates = [reranker_row["candidates"][idx] for idx in order]

    return (
        [item["text"] for item in ordered_candidates],
        [item["canonical_text"] for item in ordered_candidates],
        [item["maxfrag_text"] for item in ordered_candidates],
    )


def build_warning(row: dict[str, str]) -> str | None:
    judgment = row["focus_molecule_judgment"]
    action = row["target_action"]
    if action == "remove_focus_molecule" and judgment == "true_contributor":
        return "true_contributor paired with remove_focus_molecule"
    if action == "keep_as_is" and judgment == "non_contributing_process_molecule":
        return "non_contributing_process_molecule paired with keep_as_is"
    if action == "exclude_row" and judgment == "true_contributor":
        return "true_contributor paired with exclude_row"
    return None


def main() -> None:
    args = parse_args()
    for path, name in (
        (args.audit_csv, "audit_csv"),
        (args.test_jsonl, "test_jsonl"),
        (args.predictions_jsonl, "predictions_jsonl"),
        (args.reranker_jsonl, "reranker_jsonl"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    audit_rows = load_audit_rows(args.audit_csv)
    sample_indices = {int(row["sample_idx"]) for row in audit_rows}
    data_rows = load_selected_jsonl_rows(args.test_jsonl, sample_indices)
    prediction_rows = load_selected_jsonl_rows(args.predictions_jsonl, sample_indices)
    reranker_rows = load_selected_reranker_rows(args.reranker_jsonl, sample_indices)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    original_buckets = {name: empty_metric_bucket() for name in MODEL_NAMES}
    effective_buckets = {name: empty_metric_bucket() for name in MODEL_NAMES}
    counts = {
        "total_audit_rows": len(audit_rows),
        "pending_rows": 0,
        "included_rows": 0,
        "excluded_rows": 0,
        "keep_as_is_rows": 0,
        "remove_focus_molecule_rows": 0,
        "exclude_row_rows": 0,
        "unclear_rows": 0,
        "target_changed_rows": 0,
        "warning_rows": 0,
    }
    by_focus = {
        focus_name: {key: 0 for key in ("total", "pending", "included", "excluded", "target_changed")}
        for focus_name in FOCUS_SMILES
    }
    warnings: list[dict[str, str | int]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for audit_row in audit_rows:
            sample_idx = int(audit_row["sample_idx"])
            data_row = data_rows[sample_idx]
            prediction_row = prediction_rows[sample_idx]
            reranker_row = reranker_rows[sample_idx]

            if data_row["product"] != audit_row["product"] or data_row["target_text"] != audit_row["target_text"]:
                raise ValueError(f"row mismatch at sample_idx={sample_idx}")
            if prediction_row["product"] != audit_row["product"] or prediction_row["target_text"] != audit_row["target_text"]:
                raise ValueError(f"prediction mismatch at sample_idx={sample_idx}")
            if reranker_row["product"] != audit_row["product"] or reranker_row["target_text"] != audit_row["target_text"]:
                raise ValueError(f"reranker mismatch at sample_idx={sample_idx}")

            focus_name = audit_row["focus_molecule"]
            focus_smiles = FOCUS_SMILES[focus_name]
            action = audit_row["target_action"]
            by_focus[focus_name]["total"] += 1

            effective_target_text = apply_target_action(audit_row["target_text"], focus_smiles, action)
            evaluation_status = "pending"
            if action == "":
                counts["pending_rows"] += 1
                by_focus[focus_name]["pending"] += 1
            elif action in {"exclude_row", "unclear"}:
                counts["excluded_rows"] += 1
                counts[f"{action}_rows"] += 1
                by_focus[focus_name]["excluded"] += 1
                evaluation_status = "excluded"
            else:
                counts["included_rows"] += 1
                counts[f"{action}_rows"] += 1
                by_focus[focus_name]["included"] += 1
                evaluation_status = "included"
                if effective_target_text != audit_row["target_text"]:
                    counts["target_changed_rows"] += 1
                    by_focus[focus_name]["target_changed"] += 1

                original_canonical = canonicalize_reactants(audit_row["target_text"])
                original_maxfrag = largest_fragment(audit_row["target_text"])
                effective_canonical = canonicalize_reactants(effective_target_text)
                effective_maxfrag = largest_fragment(effective_target_text)

                for model_name in MODEL_NAMES:
                    if model_name == "before":
                        predictions, canonical_predictions, maxfrag_predictions = baseline_predictions(prediction_row)
                    else:
                        predictions, canonical_predictions, maxfrag_predictions = reranker_predictions(reranker_row, model_name)
                    update_metric_bucket(
                        original_buckets[model_name],
                        audit_row["target_text"],
                        original_canonical,
                        original_maxfrag,
                        predictions,
                        canonical_predictions,
                        maxfrag_predictions,
                    )
                    update_metric_bucket(
                        effective_buckets[model_name],
                        effective_target_text,
                        effective_canonical,
                        effective_maxfrag,
                        predictions,
                        canonical_predictions,
                        maxfrag_predictions,
                    )

            warning = build_warning(audit_row)
            if warning is not None:
                counts["warning_rows"] += 1
                warnings.append({"sample_idx": sample_idx, "warning": warning})

            output_row = {
                **audit_row,
                "effective_target_text": effective_target_text,
                "evaluation_status": evaluation_status,
                "target_changed": bool(effective_target_text is not None and effective_target_text != audit_row["target_text"]),
                "baseline_top1": prediction_row["predictions"][0] if prediction_row["predictions"] else None,
                "baseline_exact_rank_original_target": find_exact_rank(audit_row["target_text"], prediction_row["predictions"]),
                "baseline_exact_rank_effective_target": find_exact_rank(effective_target_text, prediction_row["predictions"]) if effective_target_text else None,
                "reranker_top1_by_score": {
                    score_name: reranker_row[field_name]
                    for score_name, field_name in (
                        ("mean_target_eos", "top1_after_mean_target_eos"),
                        ("mean_target", "top1_after_mean_target"),
                        ("sum_target_eos", "top1_after_sum_target_eos"),
                    )
                },
                "reranker_exact_rank_original_target": {
                    score_name: find_exact_rank(audit_row["target_text"], reranker_predictions(reranker_row, score_name)[0])
                    for score_name in ("mean_target_eos", "mean_target", "sum_target_eos")
                },
                "reranker_exact_rank_effective_target": {
                    score_name: (find_exact_rank(effective_target_text, reranker_predictions(reranker_row, score_name)[0]) if effective_target_text else None)
                    for score_name in ("mean_target_eos", "mean_target", "sum_target_eos")
                },
                "warning": warning,
            }
            fout.write(json.dumps(output_row, ensure_ascii=True) + "\n")

    metrics = {
        "audit_csv": str(args.audit_csv),
        "test_jsonl": str(args.test_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "reranker_jsonl": str(args.reranker_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "counts": counts,
        "by_focus": by_focus,
        "warnings": warnings,
        "included_rows_vs_original_target": {
            model_name: finalize_metric_bucket(original_buckets[model_name], counts["included_rows"])
            for model_name in MODEL_NAMES
        },
        "included_rows_vs_effective_target": {
            model_name: finalize_metric_bucket(effective_buckets[model_name], counts["included_rows"])
            for model_name in MODEL_NAMES
        },
        "top1_exact_delta_effective_minus_original": {
            model_name: (
                finalize_metric_bucket(effective_buckets[model_name], counts["included_rows"])["topk_exact_match"]["1"]
                - finalize_metric_bucket(original_buckets[model_name], counts["included_rows"])["topk_exact_match"]["1"]
            )
            for model_name in MODEL_NAMES
        },
    }

    with args.metrics_json.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=True)
        fout.write("\n")

    print(json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()

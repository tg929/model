#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_input.jsonl"
DEFAULT_CASES_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_cases.jsonl"
DEFAULT_AUDIT_CSV = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_audit.csv"

FOCUS = {
    "THF": {
        "smiles": "C1CCOC1",
        "sample_sizes": {"top1_hit": None, "top1_miss_top10_hit": None, "top10_miss": 40},
    },
    "Et3N": {
        "smiles": "CCN(CC)CC",
        "sample_sizes": {"top1_hit": None, "top1_miss_top10_hit": 20, "top10_miss": 30},
    },
}

CSV_FIELDS = [
    "focus_molecule",
    "sample_bucket",
    "sample_idx",
    "first_id",
    "first_year",
    "product",
    "target_text",
    "top1_prediction",
    "focus_molecule_judgment",
    "target_action",
    "root_cause_hypothesis",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample THF / Et3N audit cases and initialize the audit CSV.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def sample_bucket(row: dict) -> str:
    top1 = row["candidates"][0]["text"]
    target = row["target_text"]
    top10 = [item["text"] for item in row["candidates"]]
    if top1 == target:
        return "top1_hit"
    if target in top10:
        return "top1_miss_top10_hit"
    return "top10_miss"


def build_case(row: dict, focus_name: str, sample_bucket_name: str) -> dict:
    top10 = [item["text"] for item in row["candidates"]]
    return {
        "focus_molecule": focus_name,
        "sample_bucket": sample_bucket_name,
        "sample_idx": row["sample_idx"],
        "first_id": row["first_id"],
        "first_year": row["first_year"],
        "product": row["product"],
        "target_text": row["target_text"],
        "top1_prediction": top10[0] if top10 else None,
        "top10_predictions": top10,
    }


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")

    grouped = {
        focus_name: {"top1_hit": [], "top1_miss_top10_hit": [], "top10_miss": []}
        for focus_name in FOCUS
    }
    with args.input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            target_components = set(row["target_text"].split("."))
            bucket = sample_bucket(row)
            for focus_name, focus_cfg in FOCUS.items():
                if focus_cfg["smiles"] in target_components:
                    grouped[focus_name][bucket].append(build_case(row, focus_name, bucket))

    rng = random.Random(args.seed)
    selected = []
    for focus_name, focus_cfg in FOCUS.items():
        for bucket_name, cases in grouped[focus_name].items():
            sample_size = focus_cfg["sample_sizes"][bucket_name]
            if sample_size is None or sample_size >= len(cases):
                picked = list(cases)
            else:
                picked = rng.sample(cases, sample_size)
            selected.extend(sorted(picked, key=lambda item: item["sample_idx"]))

    selected.sort(key=lambda item: (item["focus_molecule"], item["sample_bucket"], item["sample_idx"]))
    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.cases_jsonl.open("w", encoding="utf-8") as fout:
        for row in selected:
            fout.write(json.dumps(row, ensure_ascii=True) + "\n")

    with args.audit_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "focus_molecule": row["focus_molecule"],
                    "sample_bucket": row["sample_bucket"],
                    "sample_idx": row["sample_idx"],
                    "first_id": row["first_id"],
                    "first_year": row["first_year"],
                    "product": row["product"],
                    "target_text": row["target_text"],
                    "top1_prediction": row["top1_prediction"],
                    "focus_molecule_judgment": "",
                    "target_action": "",
                    "root_cause_hypothesis": "",
                    "notes": "",
                }
            )

    summary = {
        focus_name: {bucket_name: len(grouped[focus_name][bucket_name]) for bucket_name in grouped[focus_name]}
        for focus_name in grouped
    }
    print(
        json.dumps(
            {
                "cases_jsonl": str(args.cases_jsonl),
                "audit_csv": str(args.audit_csv),
                "selected_rows": len(selected),
                "available_cases": summary,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

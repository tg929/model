#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_cases.jsonl"
DEFAULT_AUDIT_CSV = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_audit.csv"
DEFAULT_PAIR_CSV = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "retrosyn_pair_dedup_product_split.csv"
DEFAULT_RERANKER_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_scored.jsonl"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "audits" / "thf_et3n_round1_context.jsonl"

FOCUS_SMILES = {
    "THF": "C1CCOC1",
    "Et3N": "CCN(CC)CC",
}
MAP_NUM_RE = re.compile(r":\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an enriched JSONL context file for THF / Et3N audit review.")
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--reranker-jsonl", type=Path, default=DEFAULT_RERANKER_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    return parser.parse_args()


def split_components(text: str | None) -> list[str]:
    if not text:
        return []
    return [part for part in text.split(".") if part]


def canonicalize_smiles(text: str) -> str | None:
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def demap_component(mapped_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        stripped = MAP_NUM_RE.sub("", mapped_smiles)
        return canonicalize_smiles(stripped)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def demap_side(side_text: str) -> list[str]:
    items = []
    for component in split_components(side_text):
        demapped = demap_component(component)
        items.append(demapped if demapped is not None else component)
    return items


def parse_reaction_sides(raw_reaction: str | None) -> dict[str, str | list[str] | None]:
    if not raw_reaction:
        return {
            "raw_reaction": None,
            "mapped_left": None,
            "mapped_agents": None,
            "mapped_products": None,
            "demapped_left_components": [],
            "demapped_agent_components": [],
            "demapped_product_components": [],
        }

    parts = raw_reaction.split(">")
    if len(parts) != 3:
        return {
            "raw_reaction": raw_reaction,
            "mapped_left": raw_reaction,
            "mapped_agents": None,
            "mapped_products": None,
            "demapped_left_components": [],
            "demapped_agent_components": [],
            "demapped_product_components": [],
        }

    left, agents, products = parts
    return {
        "raw_reaction": raw_reaction,
        "mapped_left": left,
        "mapped_agents": agents,
        "mapped_products": products,
        "demapped_left_components": demap_side(left),
        "demapped_agent_components": demap_side(agents),
        "demapped_product_components": demap_side(products),
    }


def load_audit_rows(path: Path) -> dict[int, dict[str, str]]:
    by_sample_idx: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            by_sample_idx[int(row["sample_idx"])] = row
    return by_sample_idx


def load_pair_rows(path: Path, first_ids: set[str]) -> dict[str, dict[str, str]]:
    by_first_id: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if row["split"] != "test":
                continue
            first_id = row["first_id"]
            if first_id in first_ids:
                by_first_id[first_id] = row
    return by_first_id


def load_reranker_rows(path: Path, sample_indices: set[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            sample_idx = row["sample_idx"]
            if sample_idx in sample_indices:
                rows[sample_idx] = row
    return rows


def find_target_rank(target_text: str, predictions: list[str]) -> int | None:
    for rank, text in enumerate(predictions, start=1):
        if text == target_text:
            return rank
    return None


def build_context_row(case_row: dict, audit_row: dict[str, str] | None, pair_row: dict[str, str] | None, reranker_row: dict | None) -> dict:
    focus_smiles = FOCUS_SMILES[case_row["focus_molecule"]]
    target_components = split_components(case_row["target_text"])
    top1_components = split_components(case_row["top1_prediction"])
    top10_predictions = case_row["top10_predictions"]

    pair_context = parse_reaction_sides(pair_row["example_raw_reaction"] if pair_row else None)
    reranker_top1 = {}
    reranker_exact_rank = {}
    if reranker_row is not None:
        for score_name, field_name in (
            ("mean_target_eos", "top1_after_mean_target_eos"),
            ("mean_target", "top1_after_mean_target"),
            ("sum_target_eos", "top1_after_sum_target_eos"),
        ):
            reranker_top1[score_name] = reranker_row[field_name]
            order = reranker_row["reranked_candidate_indices"][score_name]
            ranked_texts = [reranker_row["candidates"][idx]["text"] for idx in order]
            reranker_exact_rank[score_name] = find_target_rank(case_row["target_text"], ranked_texts)

    return {
        **case_row,
        "focus_smiles": focus_smiles,
        "count": int(pair_row["count"]) if pair_row else None,
        "min_year": int(pair_row["min_year"]) if pair_row else None,
        "max_year": int(pair_row["max_year"]) if pair_row else None,
        "product_token_len": int(pair_row["product_token_len"]) if pair_row else None,
        "reactants_token_len": int(pair_row["reactants_token_len"]) if pair_row else None,
        "sequence_token_len": int(pair_row["sequence_token_len"]) if pair_row else None,
        "target_components": target_components,
        "top1_components": top1_components,
        "target_rank_in_top10": find_target_rank(case_row["target_text"], top10_predictions),
        "focus_in_target": focus_smiles in target_components,
        "focus_in_top1": focus_smiles in top1_components,
        "focus_in_any_top10_prediction": any(focus_smiles in split_components(text) for text in top10_predictions),
        "reranker_top1_by_score": reranker_top1,
        "reranker_target_rank_by_score": reranker_exact_rank,
        "audit_state": {
            "focus_molecule_judgment": audit_row["focus_molecule_judgment"] if audit_row else "",
            "target_action": audit_row["target_action"] if audit_row else "",
            "root_cause_hypothesis": audit_row["root_cause_hypothesis"] if audit_row else "",
            "notes": audit_row["notes"] if audit_row else "",
        },
        "raw_reaction_context": {
            **pair_context,
            "focus_in_demapped_left": focus_smiles in pair_context["demapped_left_components"],
            "focus_in_demapped_agents": focus_smiles in pair_context["demapped_agent_components"],
            "focus_in_demapped_products": focus_smiles in pair_context["demapped_product_components"],
        },
    }


def main() -> None:
    args = parse_args()
    for path, name in (
        (args.cases_jsonl, "cases_jsonl"),
        (args.audit_csv, "audit_csv"),
        (args.pair_csv, "pair_csv"),
        (args.reranker_jsonl, "reranker_jsonl"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    cases = []
    sample_indices: set[int] = set()
    first_ids: set[str] = set()
    with args.cases_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            cases.append(row)
            sample_indices.add(int(row["sample_idx"]))
            first_ids.add(row["first_id"])

    audit_rows = load_audit_rows(args.audit_csv)
    pair_rows = load_pair_rows(args.pair_csv, first_ids)
    reranker_rows = load_reranker_rows(args.reranker_jsonl, sample_indices)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for case_row in cases:
            sample_idx = int(case_row["sample_idx"])
            out_row = build_context_row(
                case_row=case_row,
                audit_row=audit_rows.get(sample_idx),
                pair_row=pair_rows.get(case_row["first_id"]),
                reranker_row=reranker_rows.get(sample_idx),
            )
            fout.write(json.dumps(out_row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "num_cases": len(cases),
                "matched_pair_rows": len(pair_rows),
                "matched_reranker_rows": len(reranker_rows),
                "output_jsonl": str(args.output_jsonl),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

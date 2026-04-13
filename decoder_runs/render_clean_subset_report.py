#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_JSON = (
    REPO_ROOT
    / "decoder_test_results"
    / "testall_epoch4_beamfix"
    / "audits"
    / "thf_et3n_round1_clean_subset_metrics.json"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "decoder_test_results"
    / "testall_epoch4_beamfix"
    / "audits"
    / "thf_et3n_round1_clean_subset_report.md"
)
DEFAULT_AUDIT_CSV = (
    REPO_ROOT
    / "decoder_test_results"
    / "testall_epoch4_beamfix"
    / "audits"
    / "thf_et3n_round1_audit.csv"
)
AUDIT_V1_BLOCKLIST = {"C1CCOC1", "CCN(CC)CC"}
DEFAULT_PROGRESS_EVERY = 1000

MODEL_ORDER = ("before", "mean_target_eos", "mean_target", "sum_target_eos")
MODEL_LABELS = {
    "before": "baseline_beam_order",
    "mean_target_eos": "reranker_mean_target_eos",
    "mean_target": "reranker_mean_target",
    "sum_target_eos": "reranker_sum_target_eos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a markdown report from thf_et3n_round1 clean-subset metrics "
            "with fixed reporting-caliber sections."
        )
    )
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Rewrite progress JSON every N scanned audit rows. Use 0 to disable periodic writes.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Path for incremental progress JSON. Defaults to <output-md>.progress.json.",
    )
    return parser.parse_args()


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=True)
        fout.write("\n")
    tmp.replace(path)


def render_model_table_lines(metrics: dict, section_key: str) -> list[str]:
    section = metrics[section_key]
    lines = [
        "| model | top1 exact | top3 exact | top5 exact | top10 exact | top1 canonical | top10 canonical | top1 maxfrag | top1 invalid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_name in MODEL_ORDER:
        row = section[model_name]
        exact = row["topk_exact_match"]
        canon = row["topk_canonical_match"]
        maxfrag = row["topk_maxfrag_match"]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[model_name],
                    fmt_pct(exact["1"]),
                    fmt_pct(exact["3"]),
                    fmt_pct(exact["5"]),
                    fmt_pct(exact["10"]),
                    fmt_pct(canon["1"]),
                    fmt_pct(canon["10"]),
                    fmt_pct(maxfrag["1"]),
                    fmt_pct(row["top1_invalid_smiles_rate"]),
                ]
            )
            + " |"
        )
    return lines


def count_rows_with_both_blocklist_molecules(
    audit_csv: Path,
    progress_every: int = 0,
    progress_json: Optional[Path] = None,
    progress_context: Optional[dict] = None,
) -> Tuple[int, int]:
    if not audit_csv.is_file():
        return 0, 0
    count = 0
    scanned_rows = 0
    with audit_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            scanned_rows += 1
            parts = {part for part in row.get("target_text", "").split(".") if part}
            if AUDIT_V1_BLOCKLIST.issubset(parts):
                count += 1
            if progress_json is not None and progress_every > 0 and scanned_rows % progress_every == 0:
                payload = {
                    "status": "running",
                    "stage": "scan_audit_csv",
                    "progress_every": int(progress_every),
                    "scanned_audit_rows": scanned_rows,
                    "rows_with_both_thf_et3n": count,
                }
                if progress_context:
                    payload.update(progress_context)
                write_json_atomic(progress_json, payload)
    return count, scanned_rows


def build_report(metrics: dict, both_focus_rows: int) -> str:
    counts = metrics["counts"]
    by_focus = metrics["by_focus"]
    delta = metrics["top1_exact_delta_effective_minus_original"]
    included_rows = counts["included_rows"]

    lines: list[str] = []
    lines.append("# THF / Et3N Round1 Clean-Subset Report")
    lines.append("")
    lines.append("## 1. Audit Coverage")
    lines.append("")
    lines.append(f"- audited_rows: `{counts['total_audit_rows']}`")
    lines.append(f"- pending_rows: `{counts['pending_rows']}`")
    lines.append(f"- included_rows: `{counts['included_rows']}`")
    lines.append(f"- remove_focus_molecule_rows: `{counts['remove_focus_molecule_rows']}`")
    lines.append(f"- THF included: `{by_focus['THF']['included']}`")
    lines.append(f"- Et3N included: `{by_focus['Et3N']['included']}`")
    lines.append("")
    lines.append("## 2. Original-Target Metrics (same audited rows)")
    lines.append("")
    lines.extend(render_model_table_lines(metrics, "included_rows_vs_original_target"))
    lines.append("")
    lines.append("## 3. Effective-Target Metrics (THF/Et3N removed)")
    lines.append("")
    lines.extend(render_model_table_lines(metrics, "included_rows_vs_effective_target"))
    lines.append("")
    lines.append("## 4. Top1 Exact Delta (effective - original)")
    lines.append("")
    lines.append("| model | delta_top1_exact |")
    lines.append("| --- | --- |")
    for model_name in MODEL_ORDER:
        lines.append(f"| {MODEL_LABELS[model_name]} | {fmt_pct(delta[model_name])} |")
    lines.append("")
    lines.append("## 5. Reporting Caliber (Locked)")
    lines.append("")
    lines.append("Use two parallel views in future reports instead of a single mixed number:")
    lines.append("")
    lines.append("1. Full-test official metrics on the original target definition.")
    lines.append("2. Audit-clean metrics on the effective target definition (this report) with explicit `n` and molecule scope.")
    lines.append("")
    lines.append("Required fields for each future clean-subset report:")
    lines.append("")
    lines.append(f"- sample_scope: `THF + Et3N audited subset (n={included_rows})`")
    lines.append("- action_policy: `remove_focus_molecule`")
    lines.append("- focus_judgment_policy: `non_contributing_process_molecule`")
    lines.append("- root_cause_policy: `mapping_leak`")
    lines.append("- dual metrics: `original_target_view` and `effective_target_view`")
    lines.append("")
    lines.append("## 6. Extraction Rule Fix Profile (Audit V1)")
    lines.append("")
    lines.append("Recommended extraction profile from this completed audit round:")
    lines.append("")
    lines.append("- blocklist process molecules when they are absent from demapped product: `C1CCOC1` (THF), `CCN(CC)CC` (Et3N)")
    lines.append("- apply only after atom-map-overlap precursor selection")
    lines.append("- keep the profile name explicit in artifacts: `audit_v1_fix`")
    if both_focus_rows > 0:
        lines.append(
            f"- note: `{both_focus_rows}` audited rows contain both THF and Et3N; "
            "per-row focus removal and global blocklist removal differ on those rows"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if not args.metrics_json.is_file():
        raise FileNotFoundError(f"metrics_json not found: {args.metrics_json}")
    progress_json = args.progress_json or args.output_md.with_suffix(".progress.json")
    progress_context = {
        "metrics_json": str(args.metrics_json),
        "audit_csv": str(args.audit_csv),
        "output_md": str(args.output_md),
    }
    write_json_atomic(
        progress_json,
        {
            **progress_context,
            "status": "running",
            "stage": "start",
            "progress_every": int(args.progress_every),
            "scanned_audit_rows": 0,
            "rows_with_both_thf_et3n": 0,
        },
    )

    with args.metrics_json.open("r", encoding="utf-8") as fin:
        metrics = json.load(fin)

    both_focus_rows, scanned_audit_rows = count_rows_with_both_blocklist_molecules(
        args.audit_csv,
        progress_every=args.progress_every,
        progress_json=progress_json,
        progress_context=progress_context,
    )
    report = build_report(metrics, both_focus_rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")

    summary = {
        "metrics_json": str(args.metrics_json),
        "audit_csv": str(args.audit_csv),
        "output_md": str(args.output_md),
        "included_rows": metrics["counts"]["included_rows"],
        "pending_rows": metrics["counts"]["pending_rows"],
        "progress_json": str(progress_json),
        "progress_every": int(args.progress_every),
        "scanned_audit_rows": scanned_audit_rows,
        "rows_with_both_thf_et3n": both_focus_rows,
    }
    write_json_atomic(
        progress_json,
        {
            **summary,
            "status": "completed",
            "stage": "done",
        },
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()

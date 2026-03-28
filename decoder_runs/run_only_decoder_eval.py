#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO_ROOT / "decoder_runs" / "only_decoder_650m_v1"
DEFAULT_DATA_JSONL = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "test.jsonl"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "decoder" / "weights" / "SMILES-650M-3B-Epoch1.pt"
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "decoder" / "eval_retrosyn_only_decoder.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a decoder checkpoint into a results folder and run the beam-search retrosynthesis eval."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-name", type=str, default="latest.pt")
    parser.add_argument("--label", type=str, default=None, help="Results subdirectory name under <run-dir>/results/.")
    parser.add_argument("--data-jsonl", type=Path, default=DEFAULT_DATA_JSONL)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--model-size", type=str, default="650M")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--top-ks", type=str, default="1,3,5,10")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved paths and eval command without running it.")
    return parser.parse_args()


def sample_tag(max_samples: int | None) -> str:
    return f"test{max_samples}" if max_samples is not None else "testfull"


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint_path = (run_dir / args.checkpoint_name).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not args.data_jsonl.is_file():
        raise FileNotFoundError(f"data_jsonl not found: {args.data_jsonl}")
    if not args.weight_path.is_file():
        raise FileNotFoundError(f"weight_path not found: {args.weight_path}")
    if not args.eval_script.is_file():
        raise FileNotFoundError(f"eval_script not found: {args.eval_script}")

    tag = sample_tag(args.max_samples)
    label = args.label or tag
    results_dir = run_dir / "results" / label
    results_dir.mkdir(parents=True, exist_ok=True)

    snapshot_name = f"{checkpoint_path.stem}_{tag}_snapshot.pt"
    snapshot_path = results_dir / snapshot_name
    metrics_path = results_dir / f"{tag}_metrics.json"
    predictions_path = results_dir / f"{tag}_predictions.jsonl"

    command = [
        sys.executable,
        str(args.eval_script),
        "--data-jsonl",
        str(args.data_jsonl.resolve()),
        "--checkpoint",
        str(snapshot_path),
        "--weight-path",
        str(args.weight_path.resolve()),
        "--model-size",
        args.model_size,
        "--output-json",
        str(metrics_path),
        "--predictions-jsonl",
        str(predictions_path),
        "--beam-width",
        str(args.beam_width),
        "--top-ks",
        args.top_ks,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])

    print(f"run_dir={run_dir}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"results_dir={results_dir}")
    print(f"snapshot_path={snapshot_path}")
    print(f"metrics_path={metrics_path}")
    print(f"predictions_path={predictions_path}")
    print("command=" + " ".join(command))

    if args.dry_run:
        return

    shutil.copy2(checkpoint_path, snapshot_path)
    subprocess.run(command, check=True)

    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

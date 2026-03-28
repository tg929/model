from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSONL = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "val.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "decoder_runs" / "only_decoder_650m_5epoch"
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "decoder" / "train_retrosyn_only_decoder.py"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "decoder" / "weights" / "SMILES-650M-3B-Epoch1.pt"
DEFAULT_VOCAB_PATH = REPO_ROOT / "decoder" / "vocabs" / "vocab.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chained only-decoder retrosynthesis training across per-epoch subdirectories.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--model-size", type=str, default="650M")
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--val-checks-per-epoch", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def build_train_command(
    args: argparse.Namespace,
    epoch_dir: Path,
    best_path: Path,
    resume_checkpoint: Path | None,
    epoch_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(args.train_script),
        "--train-jsonl",
        str(args.train_jsonl),
        "--val-jsonl",
        str(args.val_jsonl),
        "--output-dir",
        str(epoch_dir),
        "--weight-path",
        str(args.weight_path),
        "--model-size",
        args.model_size,
        "--vocab-path",
        str(args.vocab_path),
        "--epochs",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--max-seq-len",
        str(args.max_seq_len),
        "--full-val",
        "--val-checks-per-epoch",
        str(args.val_checks_per_epoch),
        "--save-every-steps",
        str(args.save_every_steps),
        "--best-path",
        str(best_path),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(epoch_seed),
    ]
    if args.max_train_steps is not None:
        command.extend(["--max-train-steps", str(args.max_train_steps)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.compile:
        command.append("--compile")
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return command


def initialize_output_root(args: argparse.Namespace) -> None:
    if args.start_epoch == 1:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise FileExistsError(f"Output root is not empty: {args.output_root}")
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "run_config.json").open("w", encoding="utf-8") as fout:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fout, indent=2, ensure_ascii=True)
            fout.write("\n")
        return

    if not args.output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {args.output_root}")


def resolve_initial_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.start_epoch == 1:
        return None
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint
    return args.output_root / f"epoch{args.start_epoch - 1}" / "latest.pt"


def main() -> None:
    args = parse_args()
    initialize_output_root(args)

    best_path = args.output_root / "best.pt"
    resume_checkpoint = resolve_initial_resume_checkpoint(args)

    for epoch_idx in range(args.start_epoch, args.start_epoch + args.num_epochs):
        epoch_dir = args.output_root / f"epoch{epoch_idx}"
        if epoch_dir.exists():
            raise FileExistsError(f"Epoch directory already exists: {epoch_dir}")
        epoch_seed = args.seed + epoch_idx - 1
        command = build_train_command(args, epoch_dir, best_path, resume_checkpoint, epoch_seed)
        print(
            json.dumps(
                {
                    "event": "epoch_start",
                    "epoch": epoch_idx,
                    "output_dir": str(epoch_dir),
                    "seed": epoch_seed,
                    "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
                },
                ensure_ascii=True,
            )
        )
        subprocess.run(command, check=True)
        resume_checkpoint = epoch_dir / "latest.pt"

    print(json.dumps({"event": "run_complete", "output_root": str(args.output_root), "best_path": str(best_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

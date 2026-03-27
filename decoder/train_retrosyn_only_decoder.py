from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from loadmodel_example import DEFAULT_VOCAB_PATH, load_pretrained_model


@dataclass
class OptimConfig:
    weight_decay: float
    learning_rate: float
    betas: tuple[float, float]


class JsonlRetrosynDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.offsets: List[int] = []

        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            row = json.loads(f.readline())

        source_ids = self.tokenizer.encode(row["source_text"], add_special_tokens=False)
        target_ids = self.tokenizer.encode(row["target_text"], add_special_tokens=False)
        seq = [self.tokenizer.bos_token_id] + source_ids + target_ids + [self.tokenizer.eos_token_id]
        if len(seq) > self.max_seq_len:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_seq_len={self.max_seq_len}")

        input_ids = seq[:-1]
        labels = seq[1:]
        source_len = len(source_ids)
        labels[:source_len] = [self.tokenizer.pad_token_id] * source_len
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sequence_token_len": torch.tensor(len(seq), dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    seq_lens = []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=pad_token_id))
        attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
        seq_lens.append(item["sequence_token_len"])

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "sequence_token_len": torch.stack(seq_lens, dim=0),
    }


def move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loss(model, tokenizer, dataloader: DataLoader, device: str, autocast_dtype, max_batches: int | None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.startswith("cuda") and autocast_dtype is not None)):
            _, loss, _ = model(batch["input_ids"], tokenizer, targets=batch["labels"])

        total_loss += float(loss.item())
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"loss": mean_loss, "perplexity": perplexity, "batches": total_batches}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the local decoder on single-step retrosynthesis data.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--model-size", type=str, default="650M")
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, best_val_loss: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    model, tokenizer, device = load_pretrained_model(
        weight_path=args.weight_path,
        model_size=args.model_size,
        vocab_path=args.vocab_path,
        device=args.device,
    )

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_dataset = JsonlRetrosynDataset(args.train_jsonl, tokenizer, args.max_seq_len)
    val_dataset = JsonlRetrosynDataset(args.val_jsonl, tokenizer, args.max_seq_len)

    collate = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate,
    )

    optim_cfg = OptimConfig(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    optimizer = model.configure_optimizers(optim_cfg)
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    run_config_path = args.output_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as fout:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fout, indent=2, ensure_ascii=True)
        fout.write("\n")

    global_step = 0
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        start_time = time.time()

        for batch in train_loader:
            batch = move_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.startswith("cuda") and autocast_dtype is not None)):
                _, loss, _ = model(batch["input_ids"], tokenizer, targets=batch["labels"])
                scaled_loss = loss / args.grad_accum_steps

            scaled_loss.backward()
            epoch_loss += float(loss.item())
            epoch_batches += 1

            if (epoch_batches % args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.eval_every_steps == 0:
                    val_metrics = evaluate_loss(model, tokenizer, val_loader, device, autocast_dtype, args.max_val_batches)
                    record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss_running": epoch_loss / max(epoch_batches, 1),
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                    }
                    with metrics_path.open("a", encoding="utf-8") as fout:
                        fout.write(json.dumps(record, ensure_ascii=True) + "\n")

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, global_step, best_val_loss, args)
                    save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, global_step, best_val_loss, args)

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

        if (epoch_batches % args.grad_accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_metrics = evaluate_loss(model, tokenizer, val_loader, device, autocast_dtype, args.max_val_batches)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / max(epoch_batches, 1),
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
            "epoch_seconds": round(time.time() - start_time, 2),
        }
        with metrics_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(epoch_record, ensure_ascii=True) + "\n")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, global_step, best_val_loss, args)
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, global_step, best_val_loss, args)

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    print(json.dumps({"best_val_loss": best_val_loss, "global_step": global_step}, ensure_ascii=True))


if __name__ == "__main__":
    main()

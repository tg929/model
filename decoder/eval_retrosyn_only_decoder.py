from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import RDLogger

from loadmodel_example import DEFAULT_VOCAB_PATH, load_pretrained_model


RDLogger.DisableLog("rdApp.*")


def decode_tensor(tokenizer, token_tensor: torch.Tensor) -> str:
    text = tokenizer.decode(token_tensor[0].tolist())
    text = text.replace(" ", "")
    text = text.replace("[BOS]", "")
    text = text.replace("[EOS]", "")
    text = text.replace("[SEP]", "")
    return text


def canonicalize_reactants(text: str) -> str | None:
    parts = [part for part in text.split(".") if part]
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
    parts = [part for part in text.split(".") if part]
    if not parts:
        return None

    ranked = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        ranked.append((mol.GetNumAtoms(), canonical))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def parse_top_ks(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate only-decoder retrosynthesis checkpoints with top-k exact match.")
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, default=None, help="Optional fallback pretrained weights if --checkpoint stores only model_state_dict.")
    parser.add_argument("--model-size", type=str, default="650M")
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--top-ks", type=str, default="1,3,5,10")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--length-penalty", type=float, default=0.0, help="Per-token beam-score penalty applied after generation. Default 0.0 disables it.")
    parser.add_argument("--save-every-samples", type=int, default=1000, help="Rewrite the main metrics JSON and save a milestone metrics snapshot every N processed samples.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_checkpoint_model(args: argparse.Namespace):
    weight_path = args.weight_path if args.weight_path is not None else args.checkpoint
    model, tokenizer, device = load_pretrained_model(
        weight_path=weight_path,
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


def build_metrics(
    args: argparse.Namespace,
    total: int,
    top_ks: list[int],
    exact_hits: dict[int, int],
    canonical_hits: dict[int, int],
    maxfrag_hits: dict[int, int],
    invalid_top1: int,
    completed: bool,
) -> dict:
    return {
        "num_samples": total,
        "beam_width": args.beam_width,
        "top_ks": top_ks,
        "length_penalty": args.length_penalty,
        "save_every_samples": args.save_every_samples,
        "completed": completed,
        "topk_exact_match": {str(k): exact_hits[k] / total if total else 0.0 for k in top_ks},
        "topk_canonical_match": {str(k): canonical_hits[k] / total if total else 0.0 for k in top_ks},
        "topk_maxfrag_match": {str(k): maxfrag_hits[k] / total if total else 0.0 for k in top_ks},
        "top1_invalid_smiles_rate": invalid_top1 / total if total else 0.0,
        "checkpoint": str(args.checkpoint),
        "data_jsonl": str(args.data_jsonl),
    }


def metrics_snapshot_path(path: Path, total: int) -> Path:
    return path.with_name(f"{path.stem}_up_to_{total}{path.suffix}")


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=True)
        fout.write("\n")


def main() -> None:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    if args.beam_width < max(top_ks):
        raise ValueError("beam_width must be at least as large as the largest top-k.")
    if args.save_every_samples is not None and args.save_every_samples < 1:
        raise ValueError("--save-every-samples must be at least 1.")

    model, tokenizer, device = load_checkpoint_model(args)

    total = 0
    invalid_top1 = 0
    exact_hits = {k: 0 for k in top_ks}
    canonical_hits = {k: 0 for k in top_ks}
    maxfrag_hits = {k: 0 for k in top_ks}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.predictions_jsonl is not None:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)

    predictions_fout = None
    try:
        if args.predictions_jsonl is not None:
            predictions_fout = args.predictions_jsonl.open("w", encoding="utf-8", buffering=1)

        with args.data_jsonl.open("r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                if args.max_samples is not None and line_idx >= args.max_samples:
                    break

                row = json.loads(line)
                prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(row["source_text"], add_special_tokens=False)
                prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)

                outputs = next(
                    model.beam_search_generate(
                        prefix,
                        tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        beam_width=args.beam_width,
                        temperature=0.0,
                        top_k=None,
                        rp=1.0,
                        stream=False,
                        kv_cache=False,
                        is_simulation=True,
                        linker=False,
                        num_return_sequences=args.beam_width,
                        length_penalty=args.length_penalty,
                    )
                )

                predictions = [decode_tensor(tokenizer, item) for item in outputs]
                exact_target = row["target_text"]
                canonical_target = canonicalize_reactants(exact_target)
                maxfrag_target = largest_fragment(exact_target)
                canonical_predictions = [canonicalize_reactants(pred) for pred in predictions]
                maxfrag_predictions = [largest_fragment(pred) for pred in predictions]

                total += 1
                if canonical_predictions and canonical_predictions[0] is None:
                    invalid_top1 += 1

                for k in top_ks:
                    topk_preds = predictions[:k]
                    topk_canonical = canonical_predictions[:k]
                    if exact_target in topk_preds:
                        exact_hits[k] += 1
                    if canonical_target is not None and canonical_target in topk_canonical:
                        canonical_hits[k] += 1
                    if maxfrag_target is not None and maxfrag_target in maxfrag_predictions[:k]:
                        maxfrag_hits[k] += 1

                if predictions_fout is not None:
                    predictions_fout.write(
                        json.dumps(
                            {
                                "product": row["product"],
                                "target_text": exact_target,
                                "canonical_target": canonical_target,
                                "maxfrag_target": maxfrag_target,
                                "predictions": predictions,
                                "canonical_predictions": canonical_predictions,
                                "maxfrag_predictions": maxfrag_predictions,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                if args.save_every_samples is not None and total % args.save_every_samples == 0:
                    if predictions_fout is not None:
                        predictions_fout.flush()
                        os.fsync(predictions_fout.fileno())
                    metrics = build_metrics(args, total, top_ks, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, completed=False)
                    write_metrics(args.output_json, metrics)
                    write_metrics(metrics_snapshot_path(args.output_json, total), metrics)
                    print(json.dumps({"event": "progress_save", "num_samples": total, "output_json": str(args.output_json)}, ensure_ascii=True))

        if predictions_fout is not None:
            predictions_fout.flush()
            os.fsync(predictions_fout.fileno())
    finally:
        if predictions_fout is not None:
            predictions_fout.close()

    metrics = build_metrics(args, total, top_ks, exact_hits, canonical_hits, maxfrag_hits, invalid_top1, completed=True)
    write_metrics(args.output_json, metrics)
    if args.save_every_samples is None or total % args.save_every_samples != 0:
        write_metrics(metrics_snapshot_path(args.output_json, total), metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

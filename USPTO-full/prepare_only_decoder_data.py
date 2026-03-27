from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from extract_retrosyn_data import mapped_precursors


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DECODER_ROOT = PROJECT_ROOT / "decoder"
DEFAULT_INPUT = ROOT / "uspto_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "processed_only_decoder"
DEFAULT_DECODER_VOCAB = DECODER_ROOT / "vocabs" / "vocab.txt"

sys.path.insert(0, str(PROJECT_ROOT))
from decoder.tokenizer import SmilesTokenizer  # noqa: E402


def build_decoder_tokenizer(vocab_path: Path) -> SmilesTokenizer:
    tokenizer = SmilesTokenizer(str(vocab_path))
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    tokenizer.unk_token = "[UNK]"
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
    return tokenizer


def aggregate_from_source(input_path: Path, raw_path: Path, tokenizer: SmilesTokenizer) -> tuple[int, List[dict]]:
    grouped: Dict[tuple, dict] = {}
    extracted_rows = 0
    raw_fieldnames = ["id", "patent_id", "year", "product", "reactants", "raw_reaction"]

    with input_path.open("r", newline="", encoding="utf-8") as fin, raw_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=raw_fieldnames)
        writer.writeheader()

        for source_row in reader:
            reaction_id = source_row["ID"]
            year = int(source_row["Year"])
            patent_id = reaction_id.split(";;", 1)[0]
            raw_reaction = source_row["ReactionSmiles"]
            result = mapped_precursors(raw_reaction)
            if result is None:
                continue

            product, reactants = result
            extracted_row = {
                "id": reaction_id,
                "patent_id": patent_id,
                "year": year,
                "product": product,
                "reactants": reactants,
                "raw_reaction": raw_reaction,
            }
            writer.writerow(extracted_row)
            extracted_rows += 1

            key = (product, reactants)
            bucket = grouped.get(key)
            if bucket is None:
                product_ids = tokenizer.encode(product, add_special_tokens=False)
                reactant_ids = tokenizer.encode(reactants, add_special_tokens=False)
                unk_id = tokenizer.unk_token_id

                bucket = {
                    "product": product,
                    "reactants": reactants,
                    "count": 0,
                    "first_id": reaction_id,
                    "first_patent_id": patent_id,
                    "first_year": year,
                    "min_year": year,
                    "max_year": year,
                    "example_raw_reaction": raw_reaction,
                    "product_char_len": len(product),
                    "reactants_char_len": len(reactants),
                    "product_token_len": len(product_ids),
                    "reactants_token_len": len(reactant_ids),
                    "sequence_token_len": len(product_ids) + len(reactant_ids) + 3,
                    "product_unk_count": sum(tok == unk_id for tok in product_ids),
                    "reactants_unk_count": sum(tok == unk_id for tok in reactant_ids),
                    "product_hash": hashlib.sha1(product.encode("utf-8")).hexdigest()[:16],
                }
                grouped[key] = bucket

            bucket["count"] += 1
            bucket["min_year"] = min(bucket["min_year"], year)
            bucket["max_year"] = max(bucket["max_year"], year)

    pair_rows = sorted(
        grouped.values(),
        key=lambda row: (row["product"], row["reactants"], row["first_year"], row["first_id"]),
    )
    return extracted_rows, pair_rows


def assign_product_splits(
    rows: List[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row["product"]].append(row)

    rng = random.Random(seed)
    product_groups = list(groups.items())
    rng.shuffle(product_groups)
    product_groups.sort(key=lambda item: (-len(item[1]), item[0]))

    total_rows = len(rows)
    targets = {
        "train": total_rows * train_ratio,
        "val": total_rows * val_ratio,
        "test": total_rows * test_ratio,
    }
    assigned = {"train": 0, "val": 0, "test": 0}

    for product, group_rows in product_groups:
        group_size = len(group_rows)
        split = min(
            targets.keys(),
            key=lambda name: (
                assigned[name] / targets[name] if targets[name] else float("inf"),
                assigned[name],
                name,
            ),
        )
        for row in group_rows:
            row["split"] = split
        assigned[split] += group_size


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_training_record(row: dict) -> dict:
    source = f"{row['product']}>>"
    target = row["reactants"]
    return {
        "split": row["split"],
        "product": row["product"],
        "reactants": row["reactants"],
        "source_text": source,
        "target_text": target,
        "decoder_text": f"{source}{target}",
        "count": row["count"],
        "first_id": row["first_id"],
        "first_patent_id": row["first_patent_id"],
        "first_year": row["first_year"],
        "min_year": row["min_year"],
        "max_year": row["max_year"],
        "sequence_token_len": row["sequence_token_len"],
    }


def summarize(rows: List[dict]) -> dict:
    split_counter = Counter(row["split"] for row in rows)
    products_per_split = {split: len({row["product"] for row in rows if row["split"] == split}) for split in split_counter}
    split_products = {split: {row["product"] for row in rows if row["split"] == split} for split in split_counter}
    overlap_pairs = {}
    split_names = sorted(split_products)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap_pairs[f"{left}&{right}"] = len(split_products[left] & split_products[right])
    return {
        "num_pair_rows": len(rows),
        "num_unique_products": len({row["product"] for row in rows}),
        "split_rows": dict(split_counter),
        "split_unique_products": products_per_split,
        "product_overlap_across_splits": overlap_pairs,
        "max_sequence_token_len": max(row["sequence_token_len"] for row in rows),
        "p95_sequence_token_len": sorted(row["sequence_token_len"] for row in rows)[int((len(rows) - 1) * 0.95)],
        "rows_with_product_unk": sum(row["product_unk_count"] > 0 for row in rows),
        "rows_with_reactants_unk": sum(row["reactants_unk_count"] > 0 for row in rows),
        "total_frequency_mass": sum(row["count"] for row in rows),
    }


def filter_pair_rows(rows: List[dict], max_sequence_token_len: int | None, drop_unk: bool) -> tuple[List[dict], List[dict]]:
    kept = []
    dropped = []
    for row in rows:
        reasons = []
        if drop_unk and (row["product_unk_count"] > 0 or row["reactants_unk_count"] > 0):
            reasons.append("unk_token")
        if max_sequence_token_len is not None and row["sequence_token_len"] > max_sequence_token_len:
            reasons.append("too_long")

        if reasons:
            dropped_row = dict(row)
            dropped_row["drop_reason"] = "|".join(reasons)
            dropped.append(dropped_row)
        else:
            kept.append(row)

    return kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare only-decoder retrosynthesis data with metadata and product-exclusive splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decoder-vocab", type=Path, default=DEFAULT_DECODER_VOCAB)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-sequence-token-len", type=int, default=256)
    parser.add_argument("--keep-unk", action="store_true", help="Keep rows containing [UNK] tokens instead of filtering them out.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_decoder_tokenizer(args.decoder_vocab)

    raw_path = args.output_dir / "retrosyn_with_meta.csv"
    print("extracting rows from source dataset and aggregating pairs...")
    extracted_row_count, pair_rows = aggregate_from_source(args.input, raw_path, tokenizer)
    print(f"extracted_rows={extracted_row_count}")
    print(f"wrote={raw_path}")
    filtered_pair_rows, dropped_pair_rows = filter_pair_rows(
        pair_rows,
        max_sequence_token_len=args.max_sequence_token_len,
        drop_unk=not args.keep_unk,
    )
    assign_product_splits(filtered_pair_rows, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)

    pair_fieldnames = [
        "split",
        "product",
        "reactants",
        "count",
        "first_id",
        "first_patent_id",
        "first_year",
        "min_year",
        "max_year",
        "example_raw_reaction",
        "product_char_len",
        "reactants_char_len",
        "product_token_len",
        "reactants_token_len",
        "sequence_token_len",
        "product_unk_count",
        "reactants_unk_count",
        "product_hash",
    ]
    pair_path = args.output_dir / "retrosyn_pair_dedup_product_split.csv"
    write_csv(pair_path, pair_fieldnames, filtered_pair_rows)
    print(f"wrote={pair_path}")

    if dropped_pair_rows:
        dropped_fieldnames = pair_fieldnames[1:] + ["drop_reason"]
        dropped_path = args.output_dir / "retrosyn_pair_dropped.csv"
        write_csv(dropped_path, dropped_fieldnames, dropped_pair_rows)
        print(f"wrote={dropped_path}")

    decoder_rows = [build_training_record(row) for row in filtered_pair_rows]
    decoder_fieldnames = [
        "split",
        "product",
        "reactants",
        "source_text",
        "target_text",
        "decoder_text",
        "count",
        "first_id",
        "first_patent_id",
        "first_year",
        "min_year",
        "max_year",
        "sequence_token_len",
    ]
    for split in ("train", "val", "test"):
        split_rows = [row for row in decoder_rows if row["split"] == split]
        csv_path = args.output_dir / f"{split}.csv"
        jsonl_path = args.output_dir / f"{split}.jsonl"
        write_csv(csv_path, decoder_fieldnames, split_rows)
        write_jsonl(jsonl_path, split_rows)
        print(f"wrote={csv_path}")
        print(f"wrote={jsonl_path}")

    summary = summarize(filtered_pair_rows)
    summary["num_extracted_rows"] = extracted_row_count
    summary["num_pair_rows_before_filter"] = len(pair_rows)
    summary["num_pair_rows_dropped"] = len(dropped_pair_rows)
    summary["dropped_frequency_mass"] = sum(row["count"] for row in dropped_pair_rows)
    summary["seed"] = args.seed
    summary["ratios"] = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    summary["filters"] = {
        "max_sequence_token_len": args.max_sequence_token_len,
        "drop_unk": not args.keep_unk,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=True)
        fout.write("\n")
    print(f"wrote={summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

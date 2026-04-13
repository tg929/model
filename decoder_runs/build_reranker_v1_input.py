#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "testall_best_predictions.jsonl"
DEFAULT_DATA_JSONL = REPO_ROOT / "USPTO-full" / "processed_only_decoder" / "test.jsonl"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "decoder_test_results" / "testall_epoch4_beamfix" / "reranker_v1" / "v1_reranker_input.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the sample-level JSONL input for reranker v1.")
    parser.add_argument("--predictions-jsonl", type=Path, default=DEFAULT_PREDICTIONS_JSONL)
    parser.add_argument("--data-jsonl", type=Path, default=DEFAULT_DATA_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--beam-width", type=int, default=None, help="Optionally trim each sample to the first N beam candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions_jsonl.is_file():
        raise FileNotFoundError(f"predictions_jsonl not found: {args.predictions_jsonl}")
    if not args.data_jsonl.is_file():
        raise FileNotFoundError(f"data_jsonl not found: {args.data_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with (
        args.predictions_jsonl.open("r", encoding="utf-8") as predictions_fin,
        args.data_jsonl.open("r", encoding="utf-8") as data_fin,
        args.output_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for sample_idx, (prediction_line, data_line) in enumerate(zip(predictions_fin, data_fin)):
            prediction_row = json.loads(prediction_line)
            data_row = json.loads(data_line)

            if prediction_row["product"] != data_row["product"] or prediction_row["target_text"] != data_row["target_text"]:
                raise ValueError(f"row mismatch at sample_idx={sample_idx}")

            limit = args.beam_width or len(prediction_row["predictions"])
            candidates = []
            for rank, (text, canonical_text, maxfrag_text) in enumerate(
                zip(
                    prediction_row["predictions"][:limit],
                    prediction_row["canonical_predictions"][:limit],
                    prediction_row["maxfrag_predictions"][:limit],
                ),
                start=1,
            ):
                candidates.append(
                    {
                        "rank": rank,
                        "text": text,
                        "canonical_text": canonical_text,
                        "maxfrag_text": maxfrag_text,
                    }
                )

            out_row = {
                "sample_idx": sample_idx,
                "first_id": data_row["first_id"],
                "first_year": data_row["first_year"],
                "count": data_row.get("count"),
                "sequence_token_len": data_row.get("sequence_token_len"),
                "product": data_row["product"],
                "source_text": data_row["source_text"],
                "target_text": data_row["target_text"],
                "canonical_target": prediction_row["canonical_target"],
                "maxfrag_target": prediction_row["maxfrag_target"],
                "beam_width": len(candidates),
                "candidates": candidates,
            }
            fout.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            total += 1

    print(json.dumps({"num_samples": total, "output_jsonl": str(args.output_jsonl)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

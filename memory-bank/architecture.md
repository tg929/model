# Architecture

## Overview
This directory is a model-focused workspace, not a complete application. Its core contents are:

- `encoder/`: a local BERT-style molecular encoder implementation plus assets
- `decoder/`: a GPT-style SMILES decoder implementation plus assets
- `memory-bank/`: persistent project context for future AI coding sessions
- `AGENTS.md`: root execution rules for AI agents working in this directory

The current workspace does not yet provide a first-class conditioned encoder-decoder wrapper. The two main components are present, but they are not fully integrated inside this folder.

## Root Files
- `AGENTS.md`: instructs future coding agents to read the memory bank first, work incrementally, and keep the docs updated.

## Data Workspaces
- `decoder_runs/`
  - Stores only-decoder training runs, evaluation snapshots, and reusable run/eval helpers for this repository.
  - `decoder_runs/run_only_decoder_eval.py` freezes a selected run checkpoint into a results subdirectory and invokes the existing beam-search retrosynthesis evaluator with consistent naming for snapshot, metrics, and predictions artifacts.
- `USPTO-full/`
  - Stores the correct USPTO download, mapping outputs, and retrosynthesis-extraction scripts for this repository.
  - This is the canonical local source for the current USPTO reaction data workflow.
- `.metaflow/`
  - Stores local execution metadata for the USPTO preparation workflow.
  - Currently contains records for `UsptoDataPreparationFlow`, which produced the checked-in `USPTO-full/` data artifacts.
- `有待删除/data/`
  - Stores an older incorrect dataset split (`train.csv`, `eval.csv`, `test.csv`).
  - This directory is deprecated and should not be used for future experiments unless the user explicitly asks for it.

## `encoder/`
- `encoder/encoders.py`
  - Wrapper and construction layer for encoder modules.
  - Supports a HuggingFace-backed encoder path and a local custom-BERT encoder path.
  - Builds a bundle containing the encoder, tokenizer, hidden dimension, and max sequence length.
  - Supports package-style import from the project root and script-style import from inside `encoder/`.
- `encoder/local_bert.py`
  - Defines the local BERT-style encoder stack.
  - Contains config objects, embeddings, bidirectional attention blocks, pooler, MLM head, tokenizer helpers, vocab loading, and YAML loading.
  - This file is the main implementation detail behind `LocalBertEncoder`.
  - The local tokenizer now uses explicit SMILES regex tokenization and preserves token case.
  - The local BERT forward path now normalizes common 2D attention masks into the additive mask format used by attention.
  - Bidirectional attention now actually consumes the provided attention mask during `scaled_dot_product_attention`.
- `encoder/MolEncoder-SMILES-Drug-1.2B/encoder.yaml`
  - Stores the structural config for the local encoder.
- `encoder/MolEncoder-SMILES-Drug-1.2B/vocab.txt`
  - Stores the encoder token vocabulary.
- `encoder/MolEncoder-SMILES-Drug-1.2B/checkpoint.pt`
  - Stores the local pretrained encoder checkpoint.

## `decoder/`
- `decoder/tokenizer.py`
  - Implements the decoder-side SMILES tokenizer.
  - Uses explicit regex-driven tokenization logic suitable for SMILES strings.
- `decoder/model.py`
  - Implements the GPT-style decoder.
  - Includes causal self-attention, RoPE, KV cache behavior, forward-pass loss calculation, sampling, and beam search generation.
  - No longer depends on the removed `utils.train_utils.Variable` import for standalone execution.
- `decoder/loadmodel_example.py`
  - Provides the current example entry point for loading the bundled decoder checkpoint and generating a sample SMILES string.
  - Resolves default vocab and checkpoint paths relative to its own file location, so it can be run from either the project root or the `decoder/` directory.
- `decoder/train_retrosyn_only_decoder.py`
  - Fine-tunes the local decoder on retrosynthesis-only-decoder JSONL data.
  - Expects `source_text = product>>` and `target_text = reactants`.
  - Masks the source prefix tokens from the loss and trains only on target-side next-token prediction.
  - Writes `best.pt`, `latest.pt`, `metrics.jsonl`, and `run_config.json` into the specified output directory.
- `decoder/eval_retrosyn_only_decoder.py`
  - Evaluates a retrosynthesis decoder checkpoint with beam search.
  - Reports top-k exact match, canonicalized reactant match, largest-fragment match, and top-1 invalid-SMILES rate.
- `decoder/vocabs/vocab.txt`
  - Stores the decoder vocabulary.
- `decoder/weights/SMILES-650M-3B-Epoch1.pt`
  - Stores the bundled pretrained decoder checkpoint used by the example loader.

## `USPTO-full/`
- `USPTO-full/README.md`
  - Documents the local USPTO preparation and atom-mapping workflow driven by `reaction-utils`.
- `USPTO-full/prepare_uspto_full.sh`
  - Runs `rxnutils.data.uspto.preparation_pipeline` in-place under `USPTO-full/`.
- `USPTO-full/map_uspto_full.sh`
  - Runs `rxnutils.data.mapping_pipeline` against the prepared USPTO data.
- `USPTO-full/uspto_data.csv`
  - Tab-delimited atom-mapped USPTO reaction dataset with columns `ID`, `Year`, and `ReactionSmiles`.
  - This is the current input used by the retrosynthesis extraction script.
- `USPTO-full/uspto_data_cleaned.csv`
  - Additional cleaned USPTO artifact produced during preparation; present alongside the mapped source data.
- `USPTO-full/extract_retrosyn_data.py`
  - Standalone script that reads `uspto_data.csv` and writes `retrosyn_data.csv` in the same folder.
  - Parses atom-mapped `ReactionSmiles`, keeps only single-product reactions with a parseable mapped product and at least one precursor molecule sharing atom-map IDs with the product, removes atom-mapping numbers, canonicalizes SMILES, sorts precursor components, and writes `product`, `reactants`, and `raw_reaction`.
  - The agreed task definition behind this script is single-step retrosynthesis:
    - input: `product`, as de-mapped canonical product SMILES
    - target: `reactants`, as de-mapped canonical true precursor SMILES joined by `.`
    - excluded from target: solvents, catalysts, and reagent-only molecules whose atoms do not appear in the product
    - preserved for audit/debugging: `raw_reaction`, the original atom-mapped reaction SMILES
  - The script intentionally uses the original `ReactionSmiles` field instead of `ReactionSmilesClean` because the original field still preserves left/middle/right role structure plus atom-mapping information, which is needed to decide which left-side molecules truly contributed atoms to the product.
  - `ReactionSmilesClean` is not the preferred extraction source for this task because its reactant/reagent simplification can blur the role information needed for atom-map-based precursor selection.
  - The script currently uses hard-coded input/output paths relative to its own file location and does not expose CLI arguments.
- `USPTO-full/retrosyn_data.csv`
  - Generated retrosynthesis-style extraction output from `extract_retrosyn_data.py`.
  - This file should be treated as an extraction artifact; duplicates are currently preserved unless a later step writes a deduplicated derivative.
- `USPTO-full/prepare_only_decoder_data.py`
  - Builds the current only-decoder training dataset directly from `uspto_data.csv`.
  - Reuses `mapped_precursors()` from `extract_retrosyn_data.py` to recover `product -> reactants`, then aggregates duplicate `(product, reactants)` pairs, computes decoder token statistics, drops rows with `[UNK]` or sequence length above the configured threshold, and creates product-exclusive train/val/test splits.
  - Writes decoder-ready CSV and JSONL files under `USPTO-full/processed_only_decoder/`.
- `USPTO-full/processed_only_decoder/`
  - Stores the current generated only-decoder dataset artifacts.
  - `retrosyn_with_meta.csv` keeps row-level extraction metadata including `id`, `patent_id`, `year`, and `raw_reaction`.
  - `retrosyn_pair_dedup_product_split.csv` stores deduplicated `(product, reactants)` pairs plus split assignment and token-length metadata.
  - `retrosyn_pair_dropped.csv` stores rows removed by `[UNK]` or max-sequence-length filtering.
  - `train.jsonl`, `val.jsonl`, and `test.jsonl` are the current training inputs for `decoder/train_retrosyn_only_decoder.py`.
  - `summary.json` records split sizes, product-overlap checks, token-filter counts, and sequence-length statistics.

## Current Known Gaps
- Imports in the current codebase are partly script-style, so root-level execution paths are more fragile than they should be.
- There is no formal test suite yet; validation currently depends on smoke checks and local inspection.
- The retrosynthesis extraction script does not yet accept explicit `--input` or `--output` arguments.
- There is still no conditioned encoder-decoder wrapper that consumes `product` and generates `reactants` directly.

## Environment Notes
- The preferred local environment is `retrogp`.
- If a requested workflow fails because `retrogp` is missing a dependency, install the missing package in `retrogp` before continuing.
- Heavy checkpoints are part of the local source tree and should be treated as assets, not disposable artifacts.

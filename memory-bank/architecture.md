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
- `decoder_test_results/`
  - Stores evaluation outputs, long-form analysis notes, reranker artifacts, and audit records tied to specific evaluation result directories.
  - `decoder_test_results/testall_epoch4_beamfix/` currently contains the full beam-fixed evaluation outputs for the present best only-decoder checkpoint together with:
    - `analysis_report.md` for the full-test written analysis
    - `reranker_v1/` as the planned home for the first-pass reranker input/output artifacts
    - `audits/` as the working home for structured `THF / Et3N` audit sample manifests, human-reviewed audit tables, enriched audit context JSONL, and audit-driven clean-subset evaluation artifacts
- `decoder_runs/`
  - Stores only-decoder training runs, evaluation snapshots, and reusable run/eval helpers for this repository.
  - `decoder_runs/run_only_decoder_eval.py` freezes a selected run checkpoint into a results subdirectory and invokes the existing beam-search retrosynthesis evaluator with consistent naming for snapshot, metrics, and predictions artifacts.
  - `decoder_runs/run_only_decoder_5epoch.py` orchestrates chained only-decoder training runs across per-epoch subdirectories, can start a fresh experiment from the bundled pretrained decoder weights, and can continue an existing experiment from an arbitrary `start_epoch` plus resume checkpoint while sharing one root-level `best.pt`.
  - `decoder_runs/build_reranker_v1_input.py` converts the current full-test predictions plus test-set metadata into the sample-level `JSONL` input expected by the first-pass reranker workflow.
  - `decoder_runs/build_audit_context.py` joins the sampled `THF / Et3N` audit cases with test-set metadata, example raw reactions, and reranker outputs to produce a richer JSONL review context for manual auditing.
  - `decoder_runs/eval_audited_clean_subset.py` turns a filled audit CSV into an effective-target subset JSONL plus clean-subset metrics for baseline beam order and reranker variants.
  - `decoder_runs/render_clean_subset_report.py` renders a fixed-format markdown report from the clean-subset metrics JSON, including locked reporting-caliber sections and extraction-fix profile notes.
  - `decoder_runs/render_clean_subset_report.py` also supports incremental progress snapshots via `--progress-every` and `--progress-json` while scanning large audit CSV files.
  - `decoder_runs/score_reranker_v1.py` teacher-forces each beam candidate under the current decoder checkpoint and writes per-sample reranker scores plus reranked metrics artifacts.
  - `decoder_runs/sample_thf_et3n_audit_cases.py` builds the first-pass `THF / Et3N` audit sample manifest and initializes the paired human-review CSV table.
- `USPTO-full/`
  - Stores the correct USPTO download, mapping outputs, and retrosynthesis-extraction scripts for this repository.
  - This is the canonical local source for the current USPTO reaction data workflow.
- `schneider50k/`
  - Stores local USPTO-50k benchmark split files and conversion helpers.
  - `raw_train.csv`, `raw_val.csv`, `raw_test.csv` are the unknown-class version (`class=UNK`).
  - `build_typed_splits.py` fills known reaction classes (`1..10`) by exact-key mapping against the public Retrosim `data_processed.csv`.
  - `typed_train.csv`, `typed_val.csv`, `typed_test.csv` are generated known-class outputs that preserve row order and split membership.
  - `prepare_only_decoder_data_50k.py` prepares unknown-class only-decoder training/eval files while preserving official fixed split membership (`train/val/test`) with no re-splitting.
  - `processed_only_decoder_unknown/` stores generated unknown-class only-decoder artifacts (`train/val/test` JSONL/CSV, dropped rows, summary, and progress snapshot JSON).
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
  - Supports resuming from a prior training checkpoint via `--resume-checkpoint`.
  - Supports full-validation checks scheduled by epoch fraction via `--val-checks-per-epoch` together with `--full-val`.
  - Supports lightweight model-only step snapshots via `--save-every-steps`.
  - Can write its best checkpoint either inside the epoch directory or to a shared external `--best-path`.
  - Writes `latest.pt`, `metrics.jsonl`, and `run_config.json` into the specified output directory, and writes `best.pt` either locally or to the configured shared path.
- `decoder/eval_retrosyn_only_decoder.py`
  - Evaluates a retrosynthesis decoder checkpoint with beam search.
  - Reports top-k exact match, canonicalized reactant match, largest-fragment match, and top-1 invalid-SMILES rate.
  - Now streams prediction rows directly to the requested JSONL output and periodically rewrites the main metrics JSON plus milestone metrics snapshots during long evaluations, so partial progress survives long-running jobs.
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
  - Now exposes CLI arguments for `--input`, `--output`, and extraction-policy controls.
  - Supports an optional audited leakage filter profile (`--apply-audit-v1-fix`) that drops mapped `THF` (`C1CCOC1`) and `Et3N` (`CCN(CC)CC`) from extracted reactants when those molecules are absent from the demapped product.
  - Supports additional process-molecule blocklist extension through repeated `--process-molecule-smiles`.
  - Supports incremental progress snapshots via `--progress-every` and `--progress-json` so long extraction runs can persist counters every N rows.
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
  - Can now apply the same audited leakage filter profile through `--apply-audit-v1-fix`, and can extend the blocklist via repeated `--process-molecule-smiles`.
  - Writes the active extraction-policy flags (`apply_audit_v1_fix`, `process_molecule_blocklist`) into `summary.json` for reproducibility.
  - Supports incremental progress snapshots via `--progress-every` and `--progress-json` during source aggregation, and writes completion state plus output summary path when finished.
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

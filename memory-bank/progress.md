# Progress Log

## 2026-03-26
- Initialized a vibe-coding memory setup for `/data1/ytg/model`.
- Added root `AGENTS.md`.
- Added the core `memory-bank` files:
  - `game-design-document.md`
  - `tech-stack.md`
  - `implementation-plan.md`
  - `architecture.md`
  - `progress.md`
- Documented the current project as a model subworkspace with separate `encoder/` and `decoder/` components.
- Verified local environment facts:
  - preferred environment is `fraggpt`
  - `Python 3.9.23`
  - `torch 2.7.1`
  - `transformers 4.50.3`
  - `PyYAML 6.0.2`
- Recorded the current known baseline:
  - decoder weights can load in the local environment
  - decoder can generate a SMILES sample
  - decoder can continue from a prefix
  - the current decoder example is still blocked by a dead import in the checked-in code
  - the current encoder snapshot has tokenizer and attention-mask issues
- No model code or checkpoints were modified in this step.

## 2026-03-26 Decoder Standalone Fix
- Removed the dead `utils.train_utils.Variable` import from `decoder/model.py`.
- Made `decoder/loadmodel_example.py` resolve default vocab and weight paths relative to the script file instead of the caller's current working directory.
- Verified the bundled decoder example now runs from:
  - project root via `conda run -n fraggpt python decoder/loadmodel_example.py`
  - decoder directory via `conda run -n fraggpt python loadmodel_example.py`
- Re-verified prefix continuation still works after the standalone fixes.
- Step 1 of `implementation-plan.md` is now complete.

## 2026-03-26 Retrogp Environment Validation
- Installed `transformers 4.50.3` into the `retrogp` conda environment so the original `model/decoder` code can run there.
- Verified unconditional generation in `retrogp` with `python decoder/loadmodel_example.py` from the project root.
- Verified prefix-conditioned continuation in `retrogp` using only files under `/data1/ytg/model`.
- Confirmed that the original `model` directory still does not provide a standalone external-condition encoder-decoder generation entry point.

## 2026-03-26 Root Smoke Script
- Added `decoder/test_decoder.py` as a small wrapper for decoder prefix continuation.
- The script defaults to prefix `CCO` and also accepts a prefix as the first CLI argument.

## 2026-03-26 Encoder Fixes
- Fixed encoder SMILES tokenization in `encoder/local_bert.py` by wiring `SmilesTokenizer` to the regex-based `BasicSmilesTokenizer` and restoring token/id conversion helpers.
- Updated `encoder/encoders.py` so `LocalBertEncoder` builds its tokenizer with `do_lower_case=False`, preserving uppercase chemical tokens.
- Fixed encoder attention so `BidirectionalSelfFlashAttention` actually passes `attn_mask` into `scaled_dot_product_attention`.
- Normalized encoder attention-mask handling in `BERT.forward()` so common 2D masks are converted into additive attention masks before attention.
- Fixed `encoder/encoders.py` imports so `from encoder.encoders import ...` works from the project root while preserving script-style execution compatibility.
- Validated the fixes with local smoke checks:
  - `CC(=O)OCl` now tokenizes to `['C', 'C', '(', '=', 'O', ')', 'O', 'Cl']`
  - changing the attention mask now changes encoder outputs
  - package-style import of `encoder.encoders` now succeeds
- Inspected the encoder checkpoint metadata and confirmed it is a training checkpoint containing:
  - `step`
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scaler_state_dict`
  - `loss`

## 2026-03-26 Encoder Smoke Script
- Added `encoder/test_encoder.py` as a simple local smoke test for the fixed encoder path.
- The script loads the local encoder checkpoint, tokenizes a SMILES string, builds `input_ids` plus `attention_mask`, runs a forward pass, and prints the resulting tensor shape.
- The script defaults to `CC(=O)OCl` and accepts a SMILES string as the first CLI argument.

## 2026-03-26 USPTO Dataset Workflow
- Confirmed that the older dataset under `有待删除/data/` (`train.csv`, `eval.csv`, `test.csv`) is the wrong source and should not be used for future experiments.
- Confirmed that the correct USPTO data workflow for this repository is now anchored in:
  - `USPTO-full/` for downloaded and mapped data artifacts
  - `.metaflow/UsptoDataPreparationFlow` for local execution records of the preparation pipeline
- Reviewed `USPTO-full/prepare_uspto_full.sh` and `USPTO-full/map_uspto_full.sh` and documented that they run the `rxnutils` USPTO preparation and atom-mapping pipelines in-place.
- Reviewed `USPTO-full/uspto_data.csv` and confirmed that it is a tab-delimited file with columns `ID`, `Year`, and atom-mapped `ReactionSmiles`.
- Reviewed `USPTO-full/extract_retrosyn_data.py` and confirmed that it:
  - reads `USPTO-full/uspto_data.csv`
  - writes `USPTO-full/retrosyn_data.csv`
  - discards reactions whose product side is unparsable, multi-product, or missing atom maps
  - merges reactants plus reagents, keeps only molecules whose atom-map IDs overlap with the product, removes atom-mapping, canonicalizes SMILES, sorts precursor components, and writes `product`, `reactants`, and `raw_reaction`
  - preserves duplicates and currently has no CLI arguments for overriding input or output paths
- Verified the extraction script is standalone-executable in `retrogp` with a 20-row isolated sample run:
  - `total=20 kept=11 skipped=9`
- Updated the memory-bank defaults so future experiment execution uses `retrogp` instead of `fraggpt`.
- Confirmed that `retrogp` currently has the dependencies needed for the USPTO workflow, including `rdkit` and `rxnutils`.
- Observed a long-running full extraction process `python extract_retrosyn_data.py` still writing `USPTO-full/retrosyn_data.csv`; row counts collected before that process exits should be treated as in-progress, not final.

## 2026-03-26 Retrosynthesis Task Definition
- Recovered and documented the previously agreed rationale behind `USPTO-full/extract_retrosyn_data.py`.
- Recorded the task definition for future model integration work:
  - input: canonical `product` SMILES
  - target: canonical true `reactants` SMILES joined by `.`
  - excluded from target: solvents, catalysts, and non-contributing reagents
  - retained alongside targets: original atom-mapped `raw_reaction`
- Recorded the dataset-design rationale:
  - use original `ReactionSmiles`, not `ReactionSmilesClean`
  - preserve the original left / middle / right role structure
  - use atom mapping to decide which left-side molecules truly contributed atoms to the product
  - convert only those selected precursor molecules into the decoder-facing canonical SMILES target
- Recorded that this dataset design was chosen specifically to support a future single-step retrosynthesis setup with the current local `encoder/` and `decoder/` components.

## 2026-03-27 Only-Decoder Dataset And Training Entry
- Added `USPTO-full/prepare_only_decoder_data.py` to build only-decoder retrosynthesis training data directly from `USPTO-full/uspto_data.csv`.
- Confirmed the script writes its generated artifacts to `USPTO-full/processed_only_decoder/`.
- Verified the generated dataset summary:
  - `num_extracted_rows=1705815`
  - `num_pair_rows_before_filter=633559`
  - `num_pair_rows=632053`
  - `num_pair_rows_dropped=1506`
  - `train/val/test=505641/63206/63206`
  - product overlap across splits is zero for all split pairs
  - retained rows contain zero `[UNK]` tokens on both product and reactants sides
- Confirmed that the current split logic is product-exclusive and targets an `80/10/10` row-level split by deduplicated `(product, reactants)` pairs.
- Added `decoder/train_retrosyn_only_decoder.py` as the current fine-tuning entry point for only-decoder retrosynthesis training.
- Added `decoder/eval_retrosyn_only_decoder.py` as the current beam-search evaluation entry point for top-k retrosynthesis metrics.
- Ran a real-data smoke training on `USPTO-full/processed_only_decoder/train.jsonl` and `val.jsonl`:
  - `fraggpt`, `cuda`
  - `batch_size=1`
  - `max_train_steps=10`
  - output directory: `decoder_runs/only_decoder_smoke`
  - produced `best.pt`, `latest.pt`, `metrics.jsonl`, and `run_config.json`
- Ran a second probe with a more realistic configuration:
  - `fraggpt`, `cuda`
  - `batch_size=2`
  - `grad_accum_steps=8`
  - `max_train_steps=1`
  - output directory: `decoder_runs/only_decoder_probe_bs2`
  - completed successfully without an OOM, so `batch_size=2` is a safe baseline starting point on the local `RTX 5090 32GB` machine.
- Started the first longer baseline fine-tuning run:
  - output directory: `decoder_runs/only_decoder_650m_v1`
  - pretrained initialization: `decoder/weights/SMILES-650M-3B-Epoch1.pt`
  - `fraggpt`, `cuda`
  - `batch_size=2`
  - `grad_accum_steps=8`
  - `epochs=1`
  - `eval_every_steps=1000`
  - `max_val_batches=200`
  - `num_workers=2`
- Confirmed the longer run is actively training on GPU and has already written:
  - `decoder_runs/only_decoder_650m_v1/run_config.json`
  - `decoder_runs/only_decoder_650m_v1/metrics.jsonl`
  - `decoder_runs/only_decoder_650m_v1/best.pt`
  - `decoder_runs/only_decoder_650m_v1/latest.pt`
- Recorded the current observed validation trend from the ongoing baseline run:
  - step `1000`: `val_loss=0.4139`, `val_perplexity=1.5127`
  - step `2000`: `val_loss=0.3629`, `val_perplexity=1.4375`
  - step `3000`: `val_loss=0.3224`, `val_perplexity=1.3805`
  - step `4000`: `val_loss=0.3023`, `val_perplexity=1.3529`
  - step `5000`: `val_loss=0.2915`, `val_perplexity=1.3384`

## 2026-03-27 Only-Decoder Latest Snapshot Re-Eval
- Confirmed the first long baseline run finished at `global_step=31603` with:
  - `train_loss=0.1845`
  - `val_loss=0.1982`
  - `val_perplexity=1.2193`
- Created a frozen evaluation snapshot from the final `latest.pt` at:
  - `decoder_runs/only_decoder_650m_v1/results/test-2/latest_test100_snapshot.pt`
- Re-ran the same 100-sample beam-search evaluation setup used for the earlier spot check:
  - `beam_width=10`
  - `top_ks=1,3,5,10`
  - `max_new_tokens=256`
  - `max_samples=100`
  - `device=cuda`
- Recorded the new `test-2` metrics:
  - `top1 exact/canonical = 0.10`
  - `top3 exact/canonical = 0.16`
  - `top5 exact/canonical = 0.19`
  - `top10 exact/canonical = 0.21`
  - `top1 maxfrag = 0.13`
  - `top3 maxfrag = 0.21`
  - `top5 maxfrag = 0.23`
  - `top10 maxfrag = 0.25`
  - `top1 invalid = 0.00`
- Compared with the earlier `test-1` spot check:
  - top-1 exact improved from `0.08` to `0.10`
  - top-3 exact improved from `0.12` to `0.16`
  - top-5 exact improved from `0.14` to `0.19`
  - top-10 exact improved from `0.16` to `0.21`
  - top-1 maxfrag improved from `0.11` to `0.13`
  - top-3 maxfrag improved from `0.16` to `0.21`
  - top-5 maxfrag improved from `0.20` to `0.23`
  - top-10 maxfrag improved from `0.22` to `0.25`
  - top-1 invalid stayed at `0.00`
- Confirmed a current evaluation-path constraint:
  - when `decoder/eval_retrosyn_only_decoder.py` is pointed at a full training checkpoint like `latest.pt`, it must also be given `--weight-path /data1/ytg/model/decoder/weights/SMILES-650M-3B-Epoch1.pt`
  - otherwise the script tries to treat the training checkpoint as a raw model state dict and fails during the initial load

## 2026-03-27 Reusable Decoder Eval Runner
- Added `decoder_runs/run_only_decoder_eval.py` as a reusable wrapper for long-running only-decoder evaluation jobs.
- The script now:
  - resolves the repository-relative defaults for `run-dir`, test data JSONL, bundled 650M base weights, and the existing eval entry point
  - copies a selected checkpoint such as `latest.pt` into a named `results/<label>/` snapshot file
  - invokes `decoder/eval_retrosyn_only_decoder.py` with consistent artifact naming for `*_metrics.json` and `*_predictions.jsonl`
  - forwards the common knobs for `beam_width`, `top_ks`, `max_new_tokens`, `max_samples`, `device`, and `model_size`
  - supports `--dry-run` for checking the resolved paths and final eval command without launching the heavy job
- Verified the new script with:
  - `conda run -n retrogp python decoder_runs/run_only_decoder_eval.py --run-dir /data1/ytg/model/decoder_runs/only_decoder_650m_v1 --label test-script-dryrun --max-samples 1000 --dry-run`
- Started a longer `test-3-1000` evaluation run from the final `latest.pt` snapshot under:
  - `decoder_runs/only_decoder_650m_v1/results/test-3-1000/`
  - completed with:
    - `top1 exact = 0.118`
    - `top3 exact = 0.178`
    - `top5 exact = 0.208`
    - `top10 exact = 0.225`
    - `top1 canonical = 0.120`
    - `top10 canonical = 0.231`
    - `top1 maxfrag = 0.178`
    - `top10 maxfrag = 0.291`
    - `top1 invalid = 0.005`
- Confirmed that the larger 1000-sample estimate is broadly consistent with `test-2`, but slightly stronger on every top-k exact metric:
  - `test-2 top1/top3/top5/top10 exact = 0.10 / 0.16 / 0.19 / 0.21`
  - `test-3-1000 top1/top3/top5/top10 exact = 0.118 / 0.178 / 0.208 / 0.225`

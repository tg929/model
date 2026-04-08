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

## 2026-03-27 Validation And Multi-Epoch Planning Notes
- Re-checked the current training-time validation behavior in `decoder/train_retrosyn_only_decoder.py`:
  - validation is teacher-forced loss/perplexity, not beam-search generation metrics
  - `val_loader` uses `shuffle=False`
  - `max_val_batches` truncates both mid-epoch and epoch-end validation to the same fixed prefix of the validation set
- Re-verified the processed only-decoder splits are disjoint:
  - `(product, target_text)` overlap is zero across `train`, `val`, and `test`
  - `product` overlap is also zero across all split pairs
- Ran local batch-size probes on the `RTX 5090 32GB` machine and found:
  - `batch_size=16` with `grad_accum_steps=1` is a low-risk recommended setting because it preserves the original effective batch size of `16`
  - `batch_size=32` is also feasible but changes the effective batch size and optimization behavior
  - `batch_size=64` hits CUDA OOM
- Discussed a future 5-epoch chained training workflow under `decoder_runs/only_decoder_650m_5epoch/` with:
  - one subdirectory per epoch
  - periodic step checkpoints
  - full-validation checkpoints every `1/5` epoch
  - a global `best.pt` that persists across epoch boundaries
- Fixed the current intended experiment policy before implementation:
  - `epoch1` should start from the bundled pretrained decoder weights `decoder/weights/SMILES-650M-3B-Epoch1.pt`
  - periodic `epoch-step` snapshots should prioritize lower storage as long as they remain usable for downstream evaluation and retrosynthesis metrics
  - `batch_size` is still under discussion between a conservative `16 x 1` setup and a more aggressive `32 x 1` setup
- Noted an implementation gap to address before that workflow can exist:
  - the current checked-in trainer still lacks a `--resume-checkpoint` load path even though checkpoints already save `optimizer_state_dict`, `epoch`, `step`, and `best_val_loss`

## 2026-03-27 Chained 5-Epoch Training Workflow
- Implemented the agreed chained training flow for only-decoder retrosynthesis runs.
- Updated `decoder/train_retrosyn_only_decoder.py` to support:
  - `--resume-checkpoint` for continuing from the previous epoch's `latest.pt`
  - `--full-val` plus `--val-checks-per-epoch` for scheduled full-validation passes within an epoch
  - `--save-every-steps` for periodic model-only snapshots such as `1-1000.pt`
  - `--best-path` so multiple per-epoch runs can share one root-level `best.pt`
- Added `decoder_runs/run_only_decoder_5epoch.py` to orchestrate:
  - `epoch1` from the bundled pretrained decoder weights
  - `epoch2` through `epoch5` from the previous epoch's `latest.pt`
  - one output subdirectory per epoch under a shared experiment root
  - a shared root-level `best.pt` across the full 5-epoch run
- Kept the agreed storage policy:
  - per-epoch `latest.pt` is a full training checkpoint for continuation
  - root-level `best.pt` is a full training checkpoint
  - periodic `epoch-step` snapshots are model-only weights for evaluation and downstream inference
- Verified the new training entry points with:
  - `conda run -n retrogp python -m py_compile decoder/train_retrosyn_only_decoder.py decoder_runs/run_only_decoder_5epoch.py`
  - `conda run -n retrogp python decoder/train_retrosyn_only_decoder.py --help`
  - `conda run -n retrogp python decoder_runs/run_only_decoder_5epoch.py --help`

## 2026-03-28 Continue Existing Epoch Chain
- Updated `decoder_runs/run_only_decoder_5epoch.py` so it can continue an existing experiment root instead of only creating a fresh one.
- Added support for:
  - `--start-epoch` to append later epoch directories such as `epoch6` through `epoch10`
  - `--resume-checkpoint` for explicitly choosing the checkpoint to continue from
  - automatic fallback to `epoch{start_epoch-1}/latest.pt` when continuing an existing experiment root
- Preserved the existing shared-root behavior for `best.pt`, so later appended epochs continue competing against the same global validation best.

## 2026-03-28 Git Tracking For Decoder Run Scripts
- Updated `.gitignore` so `decoder_runs/` keeps ignoring run artifacts and checkpoints while allowing top-level Python helpers such as `decoder_runs/run_only_decoder_eval.py` and `decoder_runs/run_only_decoder_5epoch.py` to be tracked by git.

## 2026-04-02 Decoder Retrosynthesis Beam-Search Diagnosis
- Reviewed the current only-decoder retrosynthesis path end-to-end without changing model code:
  - data extraction in `USPTO-full/extract_retrosyn_data.py`
  - only-decoder dataset building in `USPTO-full/prepare_only_decoder_data.py`
  - training in `decoder/train_retrosyn_only_decoder.py`
  - beam-search evaluation in `decoder/eval_retrosyn_only_decoder.py`
- Reconfirmed the current processed dataset summary:
  - `num_pair_rows=632053`
  - `train/val/test=505641/63206/63206`
  - product overlap across splits remains zero
- Rechecked the `decoder_runs/only_decoder_650m_10epoch` training chain and confirmed the teacher-forced validation best checkpoint is still the epoch-4 run:
  - `best.pt` metadata points to `epoch=4`
  - `step=126412`
  - `best_val_loss=0.11964837710439812`
- Rechecked the current 1000-sample evaluation snapshot under `decoder_test_results/test1000_epoch4_/` and confirmed the stored metrics:
  - top-1 exact `0.145`
  - top-10 exact `0.255`
- Diagnosed a major beam-search implementation issue in `decoder/model.py`:
  - the current search path hard-penalizes token ids `21`, `26`, and `32`
  - in the current decoder vocab these correspond to SMILES ring digits `2`, `3`, and `4`
- Quantified the impact of that beam-search heuristic on the existing 1000-sample predictions:
  - `557 / 1000` targets contain `2`, `3`, or `4`
  - those `557` samples currently have `0` exact matches at top-1, top-3, top-5, and top-10
  - only `1 / 1000` samples has any top-10 prediction containing `2`, `3`, or `4`
  - among the remaining `443` targets without `2/3/4`, current exact-match rates are:
    - top-1 `145/443 = 0.327`
    - top-10 `255/443 = 0.576`
- Reconfirmed a second beam-search concern:
  - the implementation applies a fixed post-hoc length penalty of `0.2` per generated token
  - this biases search toward shorter reactant strings even though exact retrosynthesis targets are often multi-fragment and longer
- Measured supporting diagnostics from the stored 1000-sample predictions:
  - average target char length `46.854`
  - average top-1 prediction char length `43.298`
  - average target fragment count `1.867`
  - average top-1 prediction fragment count `2.102`
  - allowing any known target for the same product only raises top-1 exact from `0.145` to `0.161`, so dataset multi-solution ambiguity is not the main cause of the low score
- Added a standalone project document at `/data1/ytg/model/decoder逆合成任务修复.md` that records:
  - the current only-decoder retrosynthesis pipeline
  - the beam-search diagnosis
  - why the current search heuristics conflict with the retrosynthesis task
  - the recommended repair order before further model conclusions are drawn

## 2026-04-02 Decoder Retrosynthesis Beam-Search First Fix
- Implemented the first-pass beam-search cleanup for retrosynthesis evaluation.
- Updated `decoder/model.py` to:
  - remove the hard-coded score collapse on token ids `21`, `26`, and `32`
  - expose beam-search `length_penalty` as an explicit argument with default `0.0`
  - compute generated length via `shape[1]` instead of `len(tensor)`, making the length penalty semantics explicit and correct when enabled
- Updated `decoder/eval_retrosyn_only_decoder.py` to:
  - accept `--length-penalty`
  - forward that value into `model.beam_search_generate(...)`
  - write the chosen `length_penalty` into the output metrics JSON
- Updated `decoder_runs/run_only_decoder_eval.py` to forward `--length-penalty` into the eval command so future experiment reruns can reproduce either:
  - clean default search with `0.0`
  - or any later experimental non-zero value explicitly
- Verified the code changes with:
  - `conda run -n retrogp python -m py_compile decoder/model.py decoder/eval_retrosyn_only_decoder.py decoder_runs/run_only_decoder_eval.py`
  - `conda run -n retrogp python decoder/eval_retrosyn_only_decoder.py --help`
  - `conda run -n retrogp python decoder_runs/run_only_decoder_eval.py --help`

## 2026-04-02 Decoder Eval Incremental Persistence
- Updated `decoder/eval_retrosyn_only_decoder.py` so long-running retrosynthesis evaluations no longer keep all prediction rows only in memory until process exit.
- The eval script now:
  - streams each prediction row directly into the requested `predictions_jsonl`
  - rewrites the main metrics JSON every `N` processed samples
  - writes milestone metrics snapshots named like `*_up_to_1000.json`, `*_up_to_2000.json`, and so on
  - marks metrics payloads with `completed=false/true` so partial and final outputs are distinguishable
- Added `--save-every-samples` to `decoder/eval_retrosyn_only_decoder.py` with default `1000`.
- Added the same `--save-every-samples` passthrough to `decoder_runs/run_only_decoder_eval.py`.
- Updated `memory-bank/architecture.md` to note that eval now persists partial progress during long runs.

## 2026-04-08 Full-Test Analysis Report
- Added a standalone written analysis report for the full beam-fixed retrosynthesis evaluation at:
  - `decoder_test_results/testall_epoch4_beamfix/analysis_report.md`
- Consolidated the current full-test findings from:
  - `decoder_test_results/testall_epoch4_beamfix/testall_best_metrics.json`
  - `decoder_test_results/testall_epoch4_beamfix/testall_best_predictions.jsonl`
  - `USPTO-full/processed_only_decoder/test.jsonl`
- Recorded the main evaluation conclusions:
  - full-test exact-match performance is `top1=0.4007`, `top10=0.6291`
  - `top1 maxfrag=0.4840` indicates many failures still preserve the main reactant scaffold while missing the full precursor set
  - beam search adds substantial value because many exact hits already appear in `top3/top10`, suggesting reranking is a high-return next step
  - `3+` component targets remain the dominant hard case, with performance collapsing relative to `1-2` component targets
  - low-frequency and older-patent samples are substantially harder than frequent and newer samples
- Recorded a strong data-quality signal:
  - a noticeable subset of targets includes solvent/base-like molecules such as `CCN(CC)CC`, `C1CCOC1`, and `CN(C)C=O`
  - those samples underperform the rest of the test set sharply, suggesting extraction-boundary noise rather than purely model-capacity limits
- Recommended next priorities in the report:
  - reranking over existing beam candidates
  - targeted label/data cleaning for suspicious non-contributing reactant components
  - separate treatment of `3+` component retrosynthesis targets before further epoch-scaling conclusions

## 2026-04-08 Next-Step Priorities And Cleaning List
- Added a new living discussion document at:
  - `retrosyn_next_step_priorities.md`
- Turned the earlier verbal recommendation into a persistent project record with:
  - the agreed priority order `reranker -> data cleaning -> more epochs later`
  - the quantitative justification for prioritizing reranking from the current full-test top-k gaps
  - a concrete suspicious-target review shortlist covering Et3N, THF, DMF, DIPEA, diethyl ether, ethyl acetate, ethanol, and methanol
  - per-molecule sample counts, hit rates, and example `first_id` review entries from `USPTO-full/processed_only_decoder/test.jsonl`
  - a discussion-sync section listing current agreed points, open questions, and the rule that future decisions should be synced into both this living document and `memory-bank/progress.md`
- Recorded the intended near-term discussion scope:
  - first-round audit should focus on the `P0` suspicious molecules rather than trying to solve all label-boundary cases at once
  - reranker evaluation should be framed as a pure reordering test on the existing beam candidates before any new training cycle is interpreted

## 2026-04-08 Working Discussion Decisions
- Updated `retrosyn_next_step_priorities.md` with the current working plan for the next discussion round.
- Recorded the current reranker direction:
  - `v1 reranker` should stay a clean baseline that only reorders existing beam candidates
  - use decoder teacher-forced conditional scoring with target-side length normalization
  - do not mix in handcrafted chemical priors or extra learned features in `v1`
- Recorded the current first-pass data-audit direction:
  - start the `P0` suspicious-target review from `THF` and `Et3N`
  - use them as the first two probe classes because they represent solvent-like and organic-base-like boundary failures with both high suspicion and enough sample volume
  - the suggested minimum audit slice is `50` top-1 errors plus `20` hits for each of `THF` and `Et3N`
- Added the rationale behind the current reranker scoring plan into the living document:
  - `target-side mean log-prob` is the preferred `v1` main score because it aligns with the current masked target-side training objective and removes the strongest raw length bias
  - `target-side sum log-prob` should remain as the main control score because it represents the model's raw joint conditional probability for the full candidate sequence
  - no component-count penalty, suspicious-molecule penalty, or external chemistry features should be mixed into `v1`, to keep the first reranker result interpretable and separable from data-cleaning effects
- Extended the reranker discussion record with the current stance on terminal scoring:
  - `EOS` should likely be included in the `v1` target-side mean-log-prob score
  - the main rationale is that the current decoder is trained on `target + EOS`, so including `EOS` scores complete-sequence likelihood rather than only target-prefix likelihood
  - a `no-EOS` variant should still be kept as an ablation so later comparisons can isolate how much of the reranker gain comes from modeling the stop decision itself

## 2026-04-08 Locked V1 Reranker Score Set
- Promoted the current reranker-scoring preference into an explicit locked `v1` discussion result in `retrosyn_next_step_priorities.md`.
- Fixed the `v1` score set to:
  - main score: `mean(target + EOS)`
  - first control: `mean(target)`
  - second control: `sum(target + EOS)`
- Recorded the intended interpretation:
  - main score measures average confidence over the full completed candidate sequence
  - `mean(target)` isolates the contribution of stop-position scoring
  - `sum(target + EOS)` isolates the effect of removing raw length bias
- Recorded that `v1` should no longer reopen:
  - whether `EOS` belongs in the main score
  - whether `sum(target)` should replace the chosen control set
  - whether rule penalties or extra chemistry features should be mixed into the first reranker baseline

## 2026-04-08 Reranker And Audit Implementation Details
- Extended `retrosyn_next_step_priorities.md` with the concrete implementation rules for `mean(target + EOS)`:
  - every candidate should be rescored as `candidate_text + EOS`, regardless of whether the original beam output explicitly ended with `EOS`
  - generation-truncated candidates should not get an extra handcrafted penalty in `v1`; the appended `EOS` probability is expected to penalize incomplete candidates naturally
  - scoring-overflow candidates must not be silently clipped; if they cannot be fully rescored they should be marked failed and pushed to the bottom
  - the `mean(target + EOS)` denominator is fixed to `len(target_ids) + 1`
- Reworked the first-pass `THF / Et3N` audit plan into a structured review template with fields for:
  - sample bucket
  - molecule-specific judgment
  - target action
  - root-cause hypothesis
  - notes
- Corrected the earlier informal audit sampling idea after checking actual hit counts:
  - `THF` top-1 hits are too rare for a fixed “20 hit samples” plan
  - the first-pass audit should instead use adaptive sampling over `top1_hit`, `top1_miss_top10_hit`, and `top10_miss`
- Recorded the intended writeback order:
  - first use audit results to build a clean audited subset
  - only after clear root-cause concentration should those findings be promoted into extraction-rule changes

## 2026-04-08 Reranker IO And Audit File Layout
- Locked the `v1 reranker` artifact layout in `retrosyn_next_step_priorities.md`:
  - `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_input.jsonl`
  - `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_scored.jsonl`
  - `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_metrics.json`
- Recorded the format rationale:
  - reranker input/output stay sample-level `JSONL` because beam candidates and score arrays are nested structures
  - reranker metrics stay single-object `JSON` to match the existing eval metrics style
- Locked the first-pass `THF / Et3N` audit file layout:
  - sampled case manifest: `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_cases.jsonl`
  - human annotation table: `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_audit.csv`
- Recorded the format rationale:
  - the sampled case manifest should stay `JSONL` because it carries nested context such as `top10_predictions`
  - the human audit table should be `CSV` because it is a flat manually edited annotation table
- Updated `memory-bank/architecture.md` to document `decoder_test_results/` as an explicit workspace for evaluation outputs, reranker artifacts, and audit records.

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

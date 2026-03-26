# Tech Stack

## Runtime
- Preferred Python environment: `retrogp`
- Python version in `retrogp`: `3.9.25`
- Historical secondary environment still present in the workspace: `fraggpt` (`3.9.23`)

## Core Dependencies
- `torch 2.7.1`
- `transformers 4.50.3`
- `PyYAML 6.0.2`
- `rdkit 2023.09.6`
- `rxnutils` available in `retrogp`

## Project Structure
- `encoder/`: local BERT-style molecular encoder code and assets
- `decoder/`: GPT-style SMILES decoder code and assets
- `USPTO-full/`: correct USPTO download, mapping, and retrosynthesis-extraction workspace
- `.metaflow/`: local execution records for the USPTO preparation workflow
- `有待删除/data/`: previously used incorrect dataset split files kept only as deprecated staging artifacts
- `memory-bank/`: persistent context for AI-assisted development
- `AGENTS.md`: root rules for Codex or similar coding agents

## Model Assets
- Encoder vocab size: `694`
- Decoder vocab size: `591`
- Encoder config: `24 layers / 32 heads / 2048 hidden`
- Encoder checkpoint: very large local `.pt` artifact
- Decoder checkpoint: local `.pt` artifact for the bundled `650M` model

## Working Conventions
- Use local file paths and `conda run -n retrogp ...` or the `retrogp` Python binary for commands unless a task explicitly requires another environment.
- Avoid touching large checkpoints unless the task explicitly requires it.
- Prefer smoke tests, tokenization checks, import checks, and short inference runs over heavy experimentation.
- Treat bare imports in the current codebase as a signal that some modules are script-oriented rather than package-oriented.
- If a requested workflow in `retrogp` is blocked by a missing package, install the missing dependency in `retrogp` before proceeding.

## Validation Approach
- Decoder: smoke test generation, prefix continuation, and logits calculation.
- Encoder: tokenizer sanity checks, shape checks, and masked forward-pass verification.
- USPTO data prep: verify the `rxnutils` preparation and mapping scripts, then validate retrosynthesis extraction with a small sample before launching a full run.
- Integration: only after standalone component paths are stable.

# Tech Stack

## Runtime
- Preferred Python environment: `fraggpt`
- Python version in `fraggpt`: `3.9.23`

## Core Dependencies
- `torch 2.7.1`
- `transformers 4.50.3`
- `PyYAML 6.0.2`

## Project Structure
- `encoder/`: local BERT-style molecular encoder code and assets
- `decoder/`: GPT-style SMILES decoder code and assets
- `memory-bank/`: persistent context for AI-assisted development
- `AGENTS.md`: root rules for Codex or similar coding agents

## Model Assets
- Encoder vocab size: `694`
- Decoder vocab size: `591`
- Encoder config: `24 layers / 32 heads / 2048 hidden`
- Encoder checkpoint: very large local `.pt` artifact
- Decoder checkpoint: local `.pt` artifact for the bundled `650M` model

## Working Conventions
- Use local file paths and `conda run -n fraggpt ...` or the `fraggpt` Python binary for commands.
- Avoid touching large checkpoints unless the task explicitly requires it.
- Prefer smoke tests, tokenization checks, import checks, and short inference runs over heavy experimentation.
- Treat bare imports in the current codebase as a signal that some modules are script-oriented rather than package-oriented.

## Validation Approach
- Decoder: smoke test generation, prefix continuation, and logits calculation.
- Encoder: tokenizer sanity checks, shape checks, and masked forward-pass verification.
- Integration: only after standalone component paths are stable.

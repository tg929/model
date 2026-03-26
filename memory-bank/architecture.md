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

## `encoder/`
- `encoder/encoders.py`
  - Wrapper and construction layer for encoder modules.
  - Supports a HuggingFace-backed encoder path and a local custom-BERT encoder path.
  - Builds a bundle containing the encoder, tokenizer, hidden dimension, and max sequence length.
- `encoder/local_bert.py`
  - Defines the local BERT-style encoder stack.
  - Contains config objects, embeddings, bidirectional attention blocks, pooler, MLM head, tokenizer helpers, vocab loading, and YAML loading.
  - This file is the main implementation detail behind `LocalBertEncoder`.
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
- `decoder/loadmodel_example.py`
  - Provides the current example entry point for loading the bundled decoder checkpoint and generating a sample SMILES string.
- `decoder/vocabs/vocab.txt`
  - Stores the decoder vocabulary.
- `decoder/weights/SMILES-650M-3B-Epoch1.pt`
  - Stores the bundled pretrained decoder checkpoint used by the example loader.

## Current Known Gaps
- The checked-in decoder example is not fully standalone because of a dead import in `decoder/model.py`.
- The current encoder snapshot has correctness issues around tokenizer behavior and attention-mask usage.
- Imports in the current codebase are partly script-style, so root-level execution paths are more fragile than they should be.
- There is no formal test suite yet; validation currently depends on smoke checks and local inspection.

## Environment Notes
- The preferred local environment is `fraggpt`.
- Heavy checkpoints are part of the local source tree and should be treated as assets, not disposable artifacts.

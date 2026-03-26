# Model Project Brief

## Purpose
This `model/` directory is a standalone molecular-model workspace centered on SMILES processing. It currently contains:

- a BERT-style encoder for molecular token representations
- a GPT-style decoder for SMILES generation
- large pretrained checkpoints and vocab/config assets

The goal is to make this workspace reliable for iterative AI-assisted development: easy to inspect, safe to modify, and straightforward to smoke-test.

## Current State
- The project is code-and-weights only. It has no README, packaging, or formal test suite.
- The decoder can already load pretrained weights and generate SMILES when its current import-path issue is bypassed.
- The encoder code exists, but the current snapshot has known correctness gaps around tokenizer behavior and padding-mask usage.
- There is no end-to-end conditioned wrapper in this directory yet.

## Primary User Workflows
1. Run decoder smoke tests for unconditional generation.
2. Continue generation from an existing SMILES prefix.
3. Score or inspect logits for a given SMILES string.
4. Repair and validate encoder behavior.
5. Gradually evolve the directory into a cleaner, more reusable model package.

## Project Goals
1. Establish a stable memory-bank workflow for future AI coding sessions.
2. Make decoder inference runnable without manual hacks.
3. Add repeatable smoke tests for the decoder.
4. Fix encoder tokenization and padding-mask correctness.
5. Clarify whether this workspace should remain two separate components or become a conditioned encoder-decoder package.

## Non-Goals
- Retraining large checkpoints from scratch.
- Rewriting or replacing the bundled pretrained weights.
- Turning this directory into a generic public library before core execution paths are stable.

## Success Criteria
- A future AI agent can read the memory bank and immediately understand the project.
- The decoder can be run from a documented command in the local environment.
- The encoder's current limitations are either fixed or explicitly documented with tests.
- Architectural decisions and progress remain synchronized in the memory bank.

# Implementation Plan

This plan is intentionally incremental. Do not start a later step until the user validates the current step's tests.

## Step 1: Make Decoder Standalone-Executable
Status:
- Completed on `2026-03-26`.

Goal:
- Remove the current execution blocker so the bundled decoder example can run directly from this directory.

Deliverables:
- A clean standalone decoder import path.
- A documented command for loading the `650M` checkpoint and generating a sample.

Test:
- Running the decoder example from the local project root should complete without temporary import shims.

## Step 2: Add Repeatable Decoder Smoke Tests
Status:
- Next active step.

Goal:
- Create small, fast, deterministic-enough checks for the decoder's main workflows.

Deliverables:
- A smoke path for unconditional generation.
- A smoke path for prefix continuation.
- A smoke path for logits/scoring on a provided SMILES string.

Test:
- Each smoke path has a documented command and completes successfully in the local `fraggpt` environment.

## Step 3: Fix Encoder Tokenization Correctness
Goal:
- Ensure the encoder tokenization logic preserves the intended SMILES semantics and special-token handling.

Deliverables:
- A corrected tokenizer configuration and behavior.
- A documented tokenization sanity check using a short sample such as `CC(=O)OCl`.

Test:
- The encoder tokenizer should tokenize representative SMILES strings without collapsing important tokens into `[UNK]` unexpectedly.

## Step 4: Fix Encoder Padding-Mask Behavior
Goal:
- Ensure padded positions do not contribute as normal context during encoder attention.

Deliverables:
- Correct attention-mask wiring in the encoder attention path.
- A small verification script or test that compares masked vs. unmasked padded inputs.

Test:
- Padding-aware forward passes should preserve expected tensor shapes and respect the provided attention mask.

## Step 5: Clean Up Module Boundaries
Goal:
- Make execution paths less fragile by clarifying whether modules should be used as package imports, scripts, or both.

Deliverables:
- A consistent import approach.
- Updated architecture notes for any changed boundaries.

Test:
- Documented import examples work from the project root without ad hoc path hacks where possible.

## Step 6: Decide Integration Direction
Goal:
- Choose between keeping `encoder/` and `decoder/` as separate reusable components or adding a local conditioned wrapper.

Deliverables:
- A documented decision.
- If integration is selected, a thin wrapper with a minimal smoke test.

Test:
- The chosen direction is reflected in `architecture.md`, and the new entry path is runnable.

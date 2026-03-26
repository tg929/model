# AGENTS Rules for `/data1/ytg/model`

## Start Here Every Time
1. Before planning or writing code, read these files in order:
   - `memory-bank/game-design-document.md`
   - `memory-bank/tech-stack.md`
   - `memory-bank/implementation-plan.md`
   - `memory-bank/architecture.md`
   - `memory-bank/progress.md`
2. Treat `implementation-plan.md` as the current execution plan. Do not skip ahead to a later step unless the user explicitly asks for it.
3. After each meaningful change, update `memory-bank/progress.md`.
4. If file roles, module boundaries, or execution paths change, update `memory-bank/architecture.md`.

## Project Constraints
- This directory is a SMILES model subproject, not a full product application.
- Large checkpoint files under `encoder/.../checkpoint.pt` and `decoder/.../weights/*.pt` are source assets. Do not move, rewrite, or replace them unless explicitly asked.
- Prefer the `fraggpt` conda environment for Python commands and smoke tests.
- Prefer small, repeatable smoke tests over long-running training jobs.
- Treat `encoder/` and `decoder/` as separate components unless a dedicated wrapper explicitly connects them.

## Working Style
- Keep changes modular and localized.
- Add or adjust tests together with the behavior they validate whenever practical.
- When a script is not directly runnable, fix the execution path before adding more features.
- Document real constraints and known bugs instead of hiding them.

## Current Known Issues
- This directory does not yet contain a fully integrated conditioned encoder-decoder wrapper.

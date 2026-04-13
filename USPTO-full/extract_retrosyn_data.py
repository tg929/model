import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "uspto_data.csv"
OUTPUT_PATH = ROOT / "retrosyn_data.csv"
OUTPUT_FIELDS = ["product", "reactants", "raw_reaction"]
DEFAULT_PROGRESS_EVERY = 1000
AUDIT_V1_PROCESS_MOLECULES = frozenset(
    {
        "C1CCOC1",    # THF
        "CCN(CC)CC",  # Et3N
    }
)


def remove_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def split_components(smiles_text: str) -> list[str]:
    return [part for part in smiles_text.split(".") if part]


def mapped_precursors(
    reaction_smiles: str,
    process_molecule_blocklist: Optional[set[str]] = None,
) -> Optional[Tuple[str, str]]:
    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        return None
    reactants_side, reagents_side, products_side = parts

    product_mol = Chem.MolFromSmiles(products_side)
    if product_mol is None or "." in products_side:
        return None

    product_maps = {
        atom.GetAtomMapNum()
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    if not product_maps:
        return None

    product = Chem.MolToSmiles(remove_atom_mapping(product_mol), canonical=True)
    product_components = set(split_components(product))

    precursors = []
    for smiles in filter(None, (reactants_side + "." + reagents_side).split(".")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if any(atom.GetAtomMapNum() in product_maps for atom in mol.GetAtoms()):
            precursor = Chem.MolToSmiles(remove_atom_mapping(mol), canonical=True)
            if (
                process_molecule_blocklist
                and precursor in process_molecule_blocklist
                and precursor not in product_components
            ):
                continue
            precursors.append(precursor)

    if not precursors:
        return None

    precursors.sort()
    return product, ".".join(precursors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract retrosynthesis product/reactants pairs from mapped USPTO reactions. "
            "Optional process-molecule filtering can drop known mapped leakage molecules."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--apply-audit-v1-fix",
        action="store_true",
        help=(
            "Enable the first audited leakage filter profile that removes mapped THF/Et3N "
            "when they are absent from the demapped product."
        ),
    )
    parser.add_argument(
        "--process-molecule-smiles",
        action="append",
        default=[],
        help=(
            "Add an extra demapped canonical SMILES to the process-molecule blocklist. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Rewrite progress JSON every N processed rows. Use 0 to disable periodic writes.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Path for incremental progress JSON. Defaults to <output>.progress.json.",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=True)
        fout.write("\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    process_molecule_blocklist: set[str] = set()
    if args.apply_audit_v1_fix:
        process_molecule_blocklist.update(AUDIT_V1_PROCESS_MOLECULES)
    if args.process_molecule_smiles:
        process_molecule_blocklist.update(args.process_molecule_smiles)
    progress_json = args.progress_json or args.output.with_suffix(".progress.json")

    total = 0
    kept = 0
    skipped = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", newline="", encoding="utf-8") as fin, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        write_json_atomic(
            progress_json,
            {
                "status": "running",
                "input": str(args.input),
                "output": str(args.output),
                "apply_audit_v1_fix": bool(args.apply_audit_v1_fix),
                "process_molecule_blocklist": sorted(process_molecule_blocklist),
                "progress_every": int(args.progress_every),
                "total": total,
                "kept": kept,
                "skipped": skipped,
            },
        )

        for row in reader:
            total += 1
            raw_reaction = row["ReactionSmiles"]
            result = mapped_precursors(
                raw_reaction,
                process_molecule_blocklist=process_molecule_blocklist,
            )
            if result is None:
                skipped += 1
                if args.progress_every > 0 and total % args.progress_every == 0:
                    fout.flush()
                    write_json_atomic(
                        progress_json,
                        {
                            "status": "running",
                            "input": str(args.input),
                            "output": str(args.output),
                            "apply_audit_v1_fix": bool(args.apply_audit_v1_fix),
                            "process_molecule_blocklist": sorted(process_molecule_blocklist),
                            "progress_every": int(args.progress_every),
                            "last_checkpoint_total_rows": total,
                            "total": total,
                            "kept": kept,
                            "skipped": skipped,
                        },
                    )
                continue

            product, reactants = result
            writer.writerow(
                {
                    "product": product,
                    "reactants": reactants,
                    "raw_reaction": raw_reaction,
                }
            )
            kept += 1
            if args.progress_every > 0 and total % args.progress_every == 0:
                fout.flush()
                write_json_atomic(
                    progress_json,
                    {
                        "status": "running",
                        "input": str(args.input),
                        "output": str(args.output),
                        "apply_audit_v1_fix": bool(args.apply_audit_v1_fix),
                        "process_molecule_blocklist": sorted(process_molecule_blocklist),
                        "progress_every": int(args.progress_every),
                        "last_checkpoint_total_rows": total,
                        "total": total,
                        "kept": kept,
                        "skipped": skipped,
                    },
                )

    summary = {
        "status": "completed",
        "input": str(args.input),
        "output": str(args.output),
        "apply_audit_v1_fix": bool(args.apply_audit_v1_fix),
        "process_molecule_blocklist": sorted(process_molecule_blocklist),
        "progress_every": int(args.progress_every),
        "progress_json": str(progress_json),
        "total": total,
        "kept": kept,
        "skipped": skipped,
    }
    write_json_atomic(progress_json, summary)
    print(summary)


if __name__ == "__main__":
    main()

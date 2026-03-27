from pathlib import Path
import csv
from typing import Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "uspto_data.csv"
OUTPUT_PATH = ROOT / "retrosyn_data.csv"
OUTPUT_FIELDS = ["product", "reactants", "raw_reaction"]


def remove_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol
def mapped_precursors(reaction_smiles: str) -> Optional[Tuple[str, str]]:
    reactants_side, reagents_side, products_side = reaction_smiles.split(">")

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

    precursors = []
    for smiles in filter(None, (reactants_side + "." + reagents_side).split(".")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if any(atom.GetAtomMapNum() in product_maps for atom in mol.GetAtoms()):
            precursors.append(Chem.MolToSmiles(remove_atom_mapping(mol), canonical=True))

    if not precursors:
        return None

    precursors.sort()
    return product, ".".join(precursors)


def main() -> None:
    total = 0
    kept = 0
    skipped = 0

    with INPUT_PATH.open("r", newline="", encoding="utf-8") as fin, OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in reader:
            total += 1
            raw_reaction = row["ReactionSmiles"]
            result = mapped_precursors(raw_reaction)
            if result is None:
                skipped += 1
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

    print(f"total={total} kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from encoders import LocalBertEncoder


def main():
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)OCl"
    encoder_path = ROOT / "MolEncoder-SMILES-Drug-1.2B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LocalBertEncoder(str(encoder_path)).to(device).eval()
    tokenizer = encoder.tokenizer

    token_list = tokenizer.tokenize(smiles)
    input_ids = tokenizer.encode(smiles, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = input_tensor.ne(tokenizer.pad_token_id)

    with torch.no_grad():
        sequence_output = encoder(input_tensor, attention_mask)

    print(f"device: {device}")
    print(f"smiles: {smiles}")
    print(f"tokens: {token_list}")
    print(f"input_ids: {input_ids}")
    print(f"attention_mask: {attention_mask[0].to(dtype=torch.long).tolist()}")
    print(f"sequence_output_shape: {tuple(sequence_output.shape)}")
    print(f"hidden_dim: {sequence_output.shape[-1]}")


if __name__ == "__main__":
    main()

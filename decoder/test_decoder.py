from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loadmodel_example import load_pretrained_model


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "CCO"

    model, tokenizer, device = load_pretrained_model(
        weight_path=ROOT / "weights" / "SMILES-650M-3B-Epoch1.pt",
        model_size="650M",
        vocab_path=ROOT / "vocabs" / "vocab.txt",
    )

    ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = next(
        model.generate(
            x,
            tokenizer,
            max_new_tokens=32,
            temperature=1.0,
            top_k=30,
            stream=False,
            kv_cache=True,
            is_simulation=True,
        )
    )

    full_ids = torch.cat([x[0, 1:], out[0]], dim=0)
    text = (
        tokenizer.decode(full_ids.tolist())
        .replace(" ", "")
        .replace("[BOS]", "")
        .replace("[EOS]", "")
        .replace("[SEP]", "")
    )
    print(text)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import numpy as np
import requests


def prepare(destination_path: Path = Path("data/shakespeare")) -> None:
    """Prepare the "Tiny Shakespeare" dataset."""
    destination_path.mkdir(parents=True, exist_ok=True)

    # download the tiny shakespeare dataset
    input_file_path = destination_path / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path) as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    from tokenizer import Tokenizer

    Tokenizer.train(input=input_file_path, destination=destination_path, vocab_size=128)
    tokenizer = Tokenizer(destination_path / "tokenizer.model")
    train_ids = tokenizer.encode(train_data)
    val_ids = tokenizer.encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(destination_path / "train.bin")
    val_ids.tofile(destination_path / "val.bin")


if __name__ == "__main__":
    prepare()

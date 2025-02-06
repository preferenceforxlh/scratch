from pathlib import Path
import sys
import os
import time
from functools import partial
from typing import Tuple
import torch
import numpy as np

from model import Block, LlaMA, LlaMAConfig


out_dir = "out/shakespeare"
eval_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 16
# max_iters = 600000
max_iters = 200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

block_size = 1024


def main() -> None:
    train_data, val_data = load_datasets()
    config = LlaMAConfig.from_name("baby_llama")
    config.block_size = block_size
    config.vocab_size = 100  
    model = LlaMA(config)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False)
    train(model, optimizer, train_data, val_data)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:

    iter_num = 0

    input_ids, targets = get_batch(
        train_data,
        block_size=model.config.block_size,
    )
    torch.save(input_ids, 'input.pt')
    torch.save(targets, 'target.pt')

    while True:

        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate( model, val_data)
            tmp_path = os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
            print(tmp_path)
            torch.save(model.state_dict(), tmp_path)
        t0 = time.time()

        input_ids, targets = get_batch(
            train_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        iter_num += 1

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(
            val_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_batch( data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x, y


def load_datasets(data_dir: str = "data/shakespeare") -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import os
import time
from functools import partial
import numpy as np
import torch

from generate import generate
from model import Block, LlaMA, LlaMAConfig
from tokenizer import Tokenizer
from prepare_alpaca import generate_prompt


instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 3e-3
# learning_rate = 3e-5
batch_size = 16 / devices
# batch_size = 64 / devices
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
# epoch_size = 50000  # train dataset size
epoch_size = 1024     # train dataset size
num_epochs = 1
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.0
# block_size = 512
block_size = 1024
warmup_iters = 1


def main(
    data_dir: str = "data/alpaca",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/full/alpaca",
):

    pretrained_path = "./out/shakespeare/iter-000100-ckpt.pth"

    os.makedirs(out_dir, exist_ok=True)

    print(data_dir)
    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LlaMAConfig.from_name("baby_llama")
    config.block_size = block_size
    config.vocab_size = 128  # from prepare_shakespeare_llama.py
    config.padded_vocab_size = 128

    print(pretrained_path)
    checkpoint = torch.load(pretrained_path)

    # with fabric.device:
    torch.set_default_tensor_type(torch.HalfTensor)
    model = LlaMA(config).bfloat16()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    # model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)
    # optimizer = fabric.setup_optimizers(optimizer)

    train( model, optimizer, train_data, val_data, out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, f"iter-full-ckpt.pth"))


def train(
    # fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    model.train()

    for iter_num in range(max_iters):

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(train_data)
        # with fabric.no_backward_sync(model, enabled=is_accumulating):
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        loss.backward()
            # fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # if step_count % eval_interval == 0:
            #     # val_loss = validate(fabric, model, val_data)
            #     fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            #     # fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving weights to {out_dir}")
                torch.save(model.state_dict(), os.path.join(out_dir, f"iter-save-ckpt.pth"))
                # save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("data/shakespeare/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate( model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch( val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."

    output = generate_response(model, instruction)
    print(instruction)
    print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch( data: list):
    # print(len(data))
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    # x, y = to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    print(data_dir)
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    main()

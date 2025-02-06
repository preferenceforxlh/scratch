import sys
from pathlib import Path
import os
import time
import numpy as np
import torch

from generate import generate
from lora import mark_only_lora_as_trainable, lora, lora_state_dict
from model import LlaMA, LlaMAConfig
from tokenizer import Tokenizer
from prepare_alpaca import generate_prompt


instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
# learning_rate = 3e-4
learning_rate = 3e-3
batch_size = 16
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
# max_iters = 50000 * 3 // micro_batch_size
# max_iters = 20 * 3 // micro_batch_size
max_iters = 1024 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 1024  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def main(
    data_dir: str = "data/alpaca", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "data/shakespeare/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
):
    pretrained_path = "out/shakespeare/iter-000100-ckpt.pth"
    os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LlaMAConfig.from_name("baby_llama")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LlaMA(config)
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train( model, optimizer, train_data, val_data, tokenizer_path, out_dir)
    checkpoint = lora_state_dict(model)

    torch.save(checkpoint, os.path.join(out_dir, "llama-lora-finetuned.pth"))


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch( train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        loss.backward()

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate( model, val_data, tokenizer_path)
                print(f"step {iter_num}: val loss {val_loss:.4f}")
                # barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                torch.save(checkpoint, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    # encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    # fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch( val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    instruction = "who is Hamlet "
    
    output = generate_response(model, instruction, tokenizer_path)
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
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    main()

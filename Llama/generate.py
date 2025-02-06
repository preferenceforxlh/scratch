import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
from model import LlaMA
from tokenizer import Tokenizer
from utils import lazy_load


@torch.no_grad()
def generate(
    model: LlaMA,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:

    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)
    # generate max_new_tokens tokens
    i = 0
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)
        print(f"input_t{i}: ", x.int())
        i += 1

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
) -> None:

    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        model = LlaMA.from_name('baby_llama')
        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    tokenizer = Tokenizer(tokenizer_path)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    prompt_length = encoded.size(0)
    print(model)

    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k)
        t = time.perf_counter() - t0
        model.reset_cache()
        print(y)
        print(tokenizer.vocab_size)
        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

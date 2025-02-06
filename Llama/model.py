import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from utils import find_multiple

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]

@dataclass
class LlaMAConfig:
    block_size: int = 2048
    vocab_size: int = 100
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size,64)
    
    @classmethod
    def from_name(cls,name:str) -> Self:
        return cls(**llama_configs[name])

    

llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
    "baby_llama": dict(n_layer=2, n_head=8, n_embd=128)
}

class RMSNorm(nn.Module):
    def __init__(self,size:int,dim:int = -1,eps:float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
    def forward(self,x):
        norm_x = torch.mean(x * x,dim=self.dim,keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class CasualSelfAttention(nn.Module):
    def __init__(self,config:LlaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
    
    def forward(self,
            x:torch.Tensor,
            rope:RoPECache,
            mask:MaskCache,
            max_seq_length:int,
            input_pos:Optional[torch.Tensor] = None,
            kv_cache:Optional[KVCache] = None,
        ) -> Tuple[torch.Tensor,Optional[KVCache]]:
        B,T,C = x.size()
        # get attn and split
        q,k,v = self.c_attn(x).split(self.n_embd,dim=2)
        head_size = self.n_embd // self.n_head

        q = q.view(B,T,self.n_head,head_size)
        k = k.view(B,T,self.n_head,head_size)
        v = v.view(B,T,self.n_head,head_size)

        q = apply_rope(q,rope)
        k = apply_rope(k,rope)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if kv_cache is not None:
            cache_k,cache_v = kv_cache
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1,device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k,-1,dims=2)
                cache_v = torch.roll(cache_v,-1,dims=2)
            
            k = cache_k.index_copy(2,input_pos,k)
            v = cache_v.index_copy(2,input_pos,v)
            kv_cache = (cache_k,cache_v)
        # use flash attention
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output_proj
        y = self.c_proj(y)

        return y,kv_cache

def apply_rope(x:torch.Tensor,rope:RoPECache) -> torch.Tensor:
    T = x.size(1)
    rope_cache = rope[:T]
    
    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

class MLP(nn.Module):
    def __init__(self,config:LlaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden,256)

        self.fc_1 = nn.Linear(config.n_embd,n_hidden,bias=False)
        self.fc_2 = nn.Linear(config.n_embd,n_hidden,bias=False)
        self.c_proj = nn.Linear(n_hidden,config.n_embd)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc_1(x)) * F.silu(self.fc_2(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config:LlaMAConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x:torch.Tensor,
        rope:RoPECache,
        mask:MaskCache,
        max_seq_length:int,
        input_pos:Optional[torch.Tensor] = None,
        kv_cache:Optional[KVCache] = None
    ) -> Tuple[torch.Tensor,Optional[KVCache]]:
        h,new_kv_cache =  self.attn(self.rms_1(x),rope,mask,max_seq_length,input_pos,kv_cache)
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x,new_kv_cache

class LlaMA(nn.Module):
    def __init__(self,config:LlaMAConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.padded_vocab_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd,config.padded_vocab_size,bias=False)

        self.rope_cache:Optional[RoPECache] = None
        self.mask_cache:Optional[MaskCache] = None
        self.kv_caches:List[KVCache] = []

    def _init_weights(self,module:nn.Module) -> None:
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean = 0,std = 0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
    
    def forward(
        self,
        idx:torch.Tensor,
        max_seq_length:Optional[int] = None,
        input_pos:Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor,Tuple[torch.Tensor,List[KVCache]]]:
        B,T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)
        
        if input_pos is not None:
            rope = self.rope_cache.index_select(0,input_pos)
            mask = self.mask_cache.index_select(2,input_pos)
            mask = mask[:,:,:,:max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:,:,:T,:T]
        
        x = self.transformer.wte(idx)
        # generate
        if input_pos is None: # proxy for use_cache=False
            for block in self.transformer.h:
                x,_ = block(x,rope,mask,max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B,self.config.n_head,max_seq_length,head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape,device=idx.device,dtype=x.dtype),torch.zeros(cache_shape,device=idx.device,dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
                for i,block in enumerate(self.transformer.h):
                    x,self.kv_caches[i] = block(x,rope,mask,max_seq_length,input_pos,self.kv_caches[i])
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @classmethod
    def from_name(cls,name:str) -> Self:
        return cls(LlaMAConfig.from_name(name))


    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )
    
    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)


    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            self.rope_cache = None
            self.mask_cache = None


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache
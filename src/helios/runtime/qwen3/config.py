from dataclasses import dataclass

import torch

QWEN3_4B_MODEL_ID = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151_936
    context_length: int = 40_960
    hidden_size: int = 2_560
    n_heads: int = 32
    n_layers: int = 36
    hidden_dim: int = 9_728
    head_dim: int = 128
    n_kv_heads: int = 8
    rope_base: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16


def qwen3_4b_config(dtype: torch.dtype = torch.bfloat16) -> Qwen3Config:
    return Qwen3Config(dtype=dtype)

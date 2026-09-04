import torch
from torch import nn

from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.config import Qwen3Config


class FeedForward(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.hidden_dim, bias=False, dtype=config.dtype)
        self.up = nn.Linear(config.hidden_size, config.hidden_dim, bias=False, dtype=config.dtype)
        self.down = nn.Linear(config.hidden_dim, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.scale).to(input_dtype)


def rope_parameters(config: Qwen3Config) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequencies = 1.0 / (
        config.rope_base
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    positions = torch.arange(config.context_length, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inverse_frequencies.unsqueeze(0)
    angles = torch.cat((angles, angles), dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    start_pos: int = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    _, _, sequence_length, head_dim = x.shape
    if head_dim % 2:
        raise ValueError("Qwen3 RoPE requires an even head dimension.")
    if position_ids is None:
        cos = cos[start_pos : start_pos + sequence_length].unsqueeze(0).unsqueeze(0)
        sin = sin[start_pos : start_pos + sequence_length].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return (x * cos + torch.cat((-second, first), dim=-1) * sin).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if config.n_heads % config.n_kv_heads:
            raise ValueError("The number of Q heads must be divisible by the number of KV heads.")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.group_size = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False, dtype=config.dtype)
        self.key = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False, dtype=config.dtype)
        self.value = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False, dtype=config.dtype)
        self.output = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=False, dtype=config.dtype)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        start_pos: int = 0,
        cache: KVCache | None = None,
        layer_index: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, tokens, _ = x.shape
        queries = self.query(x).view(batch_size, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        queries = apply_rope(
            self.q_norm(queries),
            cos,
            sin,
            start_pos=start_pos,
            position_ids=position_ids,
        )
        keys = apply_rope(
            self.k_norm(keys),
            cos,
            sin,
            start_pos=start_pos,
            position_ids=position_ids,
        )
        if cache is not None:
            keys, values = cache.append(layer_index, keys, values)
        context = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            enable_gqa=self.group_size > 1,
        )
        context = context.transpose(1, 2).reshape(batch_size, tokens, -1)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        start_pos: int = 0,
        cache: KVCache | None = None,
        layer_index: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x), mask, cos, sin,
            start_pos=start_pos, cache=cache, layer_index=layer_index,
            position_ids=position_ids,
        )
        return x + self.feed_forward(self.post_attention_norm(x))

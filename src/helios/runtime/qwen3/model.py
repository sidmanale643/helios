import torch
from torch import nn

from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.config import Qwen3Config
from helios.runtime.qwen3.layers import RMSNorm, TransformerBlock, rope_parameters


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
        self.output.weight = self.token_embedding.weight
        cos, sin = rope_parameters(config)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start_pos = cache.length if cache is not None else 0
        end_pos = start_pos + input_ids.shape[-1]
        if end_pos > self.config.context_length:
            raise ValueError(f"Qwen3-4B supports at most {self.config.context_length:,} tokens per request.")
        x = self.token_embedding(input_ids)
        tokens = x.shape[1]
        mask = None
        if tokens > 1:
            query_positions = torch.arange(start_pos, end_pos, device=x.device)
            key_positions = torch.arange(end_pos, device=x.device)
            mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :]
            mask = key_mask if mask is None else mask[None, None, :, :] & key_mask
        for index, block in enumerate(self.blocks):
            x = block(
                x, mask, self.cos, self.sin,
                start_pos=start_pos, cache=cache, layer_index=index,
                position_ids=position_ids,
            )
        if cache is not None:
            cache.advance(tokens)
        x = x[:, -1:, :]
        return self.output(self.final_norm(x).to(self.config.dtype))

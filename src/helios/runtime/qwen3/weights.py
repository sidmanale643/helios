import gc
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file

from helios.runtime.qwen3.model import Qwen3Model


class Qwen3Weights:

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def load_into(self, model: Qwen3Model) -> None:
        targets = self._targets(model)
        loaded: set[str] = set()
        for shard_path in self._shards():
            shard = load_file(shard_path)
            for name, weights in shard.items():
                target = targets.get(name)
                if target is None:
                    continue
                if target.shape != weights.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {name}: expected {tuple(target.shape)}, "
                        f"got {tuple(weights.shape)}."
                    )
                with torch.no_grad():
                    target.copy_(weights)
                loaded.add(name)
            del shard
            gc.collect()
        missing = set(targets) - loaded
        # Qwen ties lm_head to the token embedding, so it is already populated.
        if missing != {"lm_head.weight"}:
            names = ", ".join(sorted(missing))
            raise ValueError(f"The Qwen3 snapshot is missing required weights: {names}")

    def _shards(self) -> Iterator[Path]:
        index_path = self.snapshot_path / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as file:
                filenames = set(json.load(file)["weight_map"].values())
            for filename in sorted(filenames):
                yield self.snapshot_path / filename
            return
        yield self.snapshot_path / "model.safetensors"

    @staticmethod
    def _targets(model: Qwen3Model) -> dict[str, torch.Tensor]:
        targets = {
            "model.embed_tokens.weight": model.token_embedding.weight,
            "model.norm.weight": model.final_norm.scale,
            "lm_head.weight": model.output.weight,
        }
        for index, block in enumerate(model.blocks):
            prefix = f"model.layers.{index}"
            targets.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": block.attention.query.weight,
                    f"{prefix}.self_attn.k_proj.weight": block.attention.key.weight,
                    f"{prefix}.self_attn.v_proj.weight": block.attention.value.weight,
                    f"{prefix}.self_attn.o_proj.weight": block.attention.output.weight,
                    f"{prefix}.self_attn.q_norm.weight": block.attention.q_norm.scale,
                    f"{prefix}.self_attn.k_norm.weight": block.attention.k_norm.scale,
                    f"{prefix}.input_layernorm.weight": block.input_norm.scale,
                    f"{prefix}.mlp.gate_proj.weight": block.feed_forward.gate.weight,
                    f"{prefix}.mlp.up_proj.weight": block.feed_forward.up.weight,
                    f"{prefix}.mlp.down_proj.weight": block.feed_forward.down.weight,
                    f"{prefix}.post_attention_layernorm.weight": block.post_attention_norm.scale,
                }
            )
        return targets

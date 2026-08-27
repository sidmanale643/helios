import time

import torch

from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling


class Decoder:
    def __init__(self, model: Qwen3Model) -> None:
        self.model = model

    def generate(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        *,
        max_total_tokens: int,
    ) -> tuple[list[int], str, dict[str, float | list[float]]]:
        capacity = len(input_ids) + sampling.max_new_tokens
        if capacity > max_total_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled "
                f"limit is {max_total_tokens:,}."
            )
        token_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        # Cache capacity includes the prompt and the maximum possible continuation.
        cache = KVCache(self.model.config, capacity, device=self.device)
        generated: list[int] = []
        inter_token_seconds: list[float] = []
        finish_reason = "length"
        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            logits = self.model(token_tensor, cache=cache)[:, -1, :]
            for index in range(sampling.max_new_tokens):
                next_token = self._sample(logits, sampling)
                self._synchronize()
                elapsed = time.perf_counter() - started
                if index == 0:
                    prefill_seconds = elapsed
                token_id = next_token.item()
                if token_id == eos_token_id:
                    finish_reason = "eos"
                    break
                if index > 0:
                    inter_token_seconds.append(elapsed)
                generated.append(token_id)
                if index + 1 < sampling.max_new_tokens:
                    self._synchronize()
                    started = time.perf_counter()
                    logits = self.model(next_token, cache=cache)[:, -1, :]
        return generated, finish_reason, {
            "prefill_seconds": prefill_seconds,
            "inter_token_seconds": inter_token_seconds,
        }

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        if sampling.temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probabilities = torch.softmax(logits / sampling.temperature, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        remove = torch.cumsum(sorted_probabilities, dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(-1, sampled)

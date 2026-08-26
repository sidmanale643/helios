import re
from pathlib import Path

from tokenizers import Tokenizer as TokenizerClient


class Qwen3Tokenizer:
    _special_tokens = (
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<think>",
        "</think>",
    )
    _special_pattern = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(self, tokenizer_path: Path) -> None:
        self.client = TokenizerClient.from_file(str(tokenizer_path))
        self.special_to_id = {
            token: token_id
            for token in self._special_tokens
            if (token_id := self.client.token_to_id(token)) is not None
        }
        self.eos_token_id = self.special_to_id["<|im_end|>"]

    def encode(self, text: str) -> list[int]:
        prompt = (
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        token_ids: list[int] = []
        for part in filter(None, self._special_pattern.split(prompt)):
            if part in self.special_to_id:
                token_ids.append(self.special_to_id[part])
            else:
                token_ids.extend(self.client.encode(part).ids)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.client.decode(token_ids, skip_special_tokens=True)

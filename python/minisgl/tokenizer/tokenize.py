from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase

from .aliasing import AliasingGuideTable, build_aliasing_guide

_DEFAULT_PAGE_SIZE = 16


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, page_size: int = _DEFAULT_PAGE_SIZE) -> None:
        self.tokenizer = tokenizer
        self.page_size = page_size

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[Tuple[torch.Tensor, AliasingGuideTable]]:
        results: List[Tuple[torch.Tensor, AliasingGuideTable]] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            ids = input_ids.view(-1).to(torch.int32)
            guide = build_aliasing_guide(ids, self.page_size)
            results.append((ids, guide))
        return results

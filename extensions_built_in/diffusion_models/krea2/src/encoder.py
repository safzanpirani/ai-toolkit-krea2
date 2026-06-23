"""Krea 2 text conditioner: Qwen3-VL-4B-Instruct, multi-layer hidden-state tap.

Vendored from the K2 reference with two ai-toolkit-friendly tweaks:
  - a ``.device`` property (ai-toolkit checks ``text_encoder.device``);
  - ``drop_visual()`` to free the unused VL vision tower (a Conv3d patch-embed
    with no fast bf16 path), saving VRAM -- text-only encoding never touches it.

The conditioner returns the K2 conditioning tensor: the Qwen3-VL hidden states
from 12 selected layers, stacked -> (B, L, 12, 2560), plus the (B, L) mask.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)


@dataclass
class TextEncoderConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        torch_dtype=None,
    ):
        super().__init__()
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            version, torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(version, max_length=max_length)
        self.processor = Qwen2TokenizerFast.from_pretrained(
            version, max_length=max_length
        )
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    @property
    def device(self):
        return self.qwen.device

    def drop_visual(self):
        """Free the VL vision tower; text-only encoding doesn't use it."""
        for owner, attr in ((getattr(self.qwen, "model", None), "visual"),
                            (self.qwen, "visual")):
            if owner is not None and getattr(owner, attr, None) is not None:
                setattr(owner, attr, None)

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(
            self.qwen.device, non_blocking=True
        )
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length
                + prefix_idx
                - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(
                input_ids=input_ids, attention_mask=mask, output_hidden_states=True
            )

            hiddens = torch.stack(
                [states.hidden_states[i] for i in self.select_layers], dim=2
            )
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as T


class CollateImageText:
    """
    Collate:
      - image preprocessing -> pixel_values [B,3,H,W] in [0,1] float32
      - tokenization -> input_ids [B,T], attention_mask [B,T]
      - labels -> [B,T] with prompt tokens masked to -100 (loss only on report tokens)
    """

    def __init__(
        self,
        tokenizer,
        prompt: str,
        image_size: int = 224,
        max_text_len: int = 512,
        max_prompt_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_text_len = max_text_len
        self.max_prompt_len = max_prompt_len

        # IMPORTANT: keep images un-normalized here (no mean/std).
        # We output [0,1] and let the vision encoder wrapper normalize once.
        self.img_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),  # -> float32 in [0,1], shape [3,H,W]
        ])

    def _as_pil(self, x: Any) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        # if dataset already returns torch tensor [3,H,W] in [0,1], convert carefully
        if torch.is_tensor(x):
            if x.ndim == 3 and x.shape[0] == 3:
                # tensor -> PIL via torchvision
                to_pil = T.ToPILImage()
                return to_pil(x).convert("RGB")
            raise ValueError(f"Unsupported tensor image shape: {tuple(x.shape)}")
        raise ValueError(f"Unsupported image type: {type(x)}")

    def _get_fields(self, ex: Any) -> Dict[str, Any]:
        """
        Supports examples that are:
          - dict with keys like: image/img, text/report/target, path
          - tuple: (image, text) or (image, text, path)
        """
        if isinstance(ex, dict):
            img = ex.get("image", ex.get("img", None))
            txt = ex.get("text", ex.get("report", ex.get("target", "")))
            path = ex.get("path", ex.get("image_path", ""))
            return {"image": img, "text": txt, "path": path}

        if isinstance(ex, (tuple, list)):
            if len(ex) == 2:
                return {"image": ex[0], "text": ex[1], "path": ""}
            if len(ex) >= 3:
                return {"image": ex[0], "text": ex[1], "path": ex[2]}
        raise ValueError(f"Unsupported example type: {type(ex)}")

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        # ---- images ----
        pixel_values = []
        paths = []
        raw_targets = []

        for ex in batch:
            f = self._get_fields(ex)
            pil = self._as_pil(f["image"])
            pixel_values.append(self.img_tf(pil))  # [3,H,W] in [0,1]
            paths.append(f["path"])
            raw_targets.append(f["text"])

        pixel_values = torch.stack(pixel_values, dim=0)  # [B,3,H,W]

        # ---- tokens (prompt + report) ----
        # We build:
        #   input_ids  = [prompt_ids, report_ids]
        #   labels     = [-100 for prompt_ids, report_ids]
        # Then pad across batch.

        input_ids_list = []
        labels_list = []
        attn_list = []

        # tokenized prompt once (same for all samples)
        prompt_ids = self.tokenizer(
            self.prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors=None,
        )["input_ids"]


        for report_text in raw_targets:
            # tokenize report
            report_ids = self.tokenizer(
                report_text,
                add_special_tokens=False,
                truncation=False,  # we control truncation manually
                return_tensors=None,
            )["input_ids"]

            # Add EOS to report if available (helps causal LM learn “end”)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                report_ids = report_ids + [eos_id]

            # Truncate to max_text_len while preserving ALL prompt tokens.
            # Keep prompt; truncate report tail if needed.
            max_report_len = max(0, self.max_text_len - len(prompt_ids))
            if len(report_ids) > max_report_len:
                report_ids = report_ids[:max_report_len]

            ids = prompt_ids + report_ids

            # Labels: prompt masked out, report supervised
            labels = ([-100] * len(prompt_ids)) + report_ids

            # Attention mask: 1 for real tokens (padding handled later)
            attn = [1] * len(ids)

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))

        # ---- pad to batch max length ----
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)

        return {
            "images": pixel_values,          # [B,3,H,W] float
            "input_ids": input_ids,                # [B,T] long
            "attention_mask": attention_mask,      # [B,T] long
            "labels": labels,                      # [B,T] long (-100 masks prompt + padding)
            "paths": paths,
            "raw_targets": raw_targets,
            "prompt": self.prompt,
        }
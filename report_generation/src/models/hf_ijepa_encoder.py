# src/models/hf_ijepa_encoder.py
import torch
import torch.nn as nn


class FrozenHFIJEPAEncoder(nn.Module):
    """
    Frozen HF iJEPA encoder wrapper.

    Contract:
      input:  images [B,3,H,W] float in [0,1]
      output: patch_tokens [B,N,D] (CLS removed if present)
    """

    def __init__(self, hf_model, image_processor=None):
        super().__init__()
        self.model = hf_model
        self.processor = image_processor

        # freeze
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = getattr(self.model, "config", None)
        self.embed_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
        )
        if self.embed_dim is None:
            raise ValueError("Could not infer embed_dim from model config.")

        # cache mean/std like you do for DINO
        if self.processor is not None:
            mean = getattr(self.processor, "image_mean", None)
            std = getattr(self.processor, "image_std", None)
            if mean is not None and std is not None:
                self.register_buffer(
                    "_mean_buf",
                    torch.tensor(mean).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "_std_buf",
                    torch.tensor(std).view(1, 3, 1, 1),
                    persistent=False,
                )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, "_mean_buf", None)
        std = getattr(self, "_std_buf", None)
        if mean is None or std is None:
            return images
        return (images - mean.to(images.device)) / std.to(images.device)

    def _strip_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B,L,D]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected [B,L,D], got {tuple(tokens.shape)}")
        # for ViT-style models, CLS is usually token 0
        return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        images = self._normalize(images)

        out = self.model(pixel_values=images, return_dict=True)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            raise RuntimeError("iJEPA model did not return last_hidden_state.")
        return self._strip_cls(tokens)
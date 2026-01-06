# src/models/rad_jepa_encoder.py
import torch
import torch.nn as nn


class FrozenRadJepaEncoder(nn.Module):
    """
    Frozen RAD-JEPA encoder wrapper.

    Contract:
      input:  images [B,3,H,W] float in [0,1]
      output: patch_tokens [B,N,D] (CLS removed if present)
    """

    def __init__(self, rad_jepa_model, image_processor=None):
        super().__init__()
        self.model = rad_jepa_model
        self.processor = image_processor

        # freeze
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # cache mean/std for fast tensor normalization if processor provided
        # IMPORTANT: don't use names "mean"/"std" as normal attrs AND buffers.
        self._mean = None
        self._std = None
        if self.processor is not None:
            mean = getattr(self.processor, "image_mean", None)
            std = getattr(self.processor, "image_std", None)
            if mean is not None and std is not None:
                self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
                self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # expose embedding dim for projector/generator
        cfg = getattr(self.model, "config", None)
        self.embed_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
            or 768
        )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """images float in [0,1], shape [B,3,H,W]"""
        if self._mean is None or self._std is None:
            return images
        return (images - self._mean.to(images.device)) / self._std.to(images.device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        images = self._normalize(images)

        # ---- Path A: HuggingFace ViT-style forward ----
        try:
            out = self.model(pixel_values=images, return_dict=True)

            tokens = getattr(out, "last_hidden_state", None)
            if tokens is not None:
                if tokens.ndim != 3:
                    raise RuntimeError(f"Unexpected last_hidden_state shape: {tuple(tokens.shape)}")
                # drop CLS if present (assume first token is CLS when length > 1)
                return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

        except TypeError:
            # signature mismatch; fall through
            pass

        # ---- Path B: custom extract_features API ----
        if hasattr(self.model, "extract_features"):
            out = self.model.extract_features(images)

            if isinstance(out, dict):
                for key in ["patch_tokens", "patchtokens", "x_norm_patchtokens", "tokens", "feat", "last_hidden_state"]:
                    if key in out:
                        pt = out[key]
                        if pt.ndim == 2:
                            return pt.unsqueeze(1)  # [B,1,D] fallback
                        if pt.ndim == 3:
                            return pt[:, 1:, :] if pt.shape[1] > 1 else pt
                        raise RuntimeError(f"Unexpected token tensor shape for key={key}: {tuple(pt.shape)}")

            if isinstance(out, (tuple, list)) and len(out) > 0:
                pt = out[0]
                if pt.ndim == 2:
                    return pt.unsqueeze(1)
                if pt.ndim == 3:
                    return pt[:, 1:, :] if pt.shape[1] > 1 else pt
                raise RuntimeError(f"Unexpected token tensor shape from extract_features: {tuple(pt.shape)}")

        raise RuntimeError(
            "Could not extract patch tokens from RAD-JEPA. "
            "Model output didn't match HF pixel_values or extract_features patterns."
        )
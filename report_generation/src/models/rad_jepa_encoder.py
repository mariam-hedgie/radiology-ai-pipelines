import torch
import torch.nn as nn


class FrozenRadJepaEncoder(nn.Module):
    """
    Frozen RAD-JEPA encoder wrapper.

    Contract (must match your generator):
      input:  images [B,3,H,W] float in [0,1]
      output: patch_tokens [B,N,D]  (preferably patches only; CLS removed when present)

    Works with:
    - HuggingFace ViT-style: model(pixel_values=...) -> outputs.last_hidden_state
    - Or custom models exposing extract_features(...)
    """

    def __init__(self, rad_jepa_model, image_processor=None):
        super().__init__()
        self.model = rad_jepa_model
        self.processor = image_processor

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # cache mean/std for fast tensor normalization if processor provided
        self.mean = None
        self.std = None
        if self.processor is not None:
            mean = getattr(self.processor, "image_mean", None)
            std = getattr(self.processor, "image_std", None)
            if mean is not None and std is not None:
                self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
                self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # expose embedding dim if available (helps projector later)
        self.embed_dim = getattr(getattr(self.model, "config", None), "hidden_size", None)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        images assumed float in [0,1], shape [B,3,H,W]
        """
        if self.mean is None or self.std is None:
            return images
        return (images - self.mean.to(images.device)) / self.std.to(images.device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        images = self._normalize(images)

        # ---- Path A: HuggingFace ViT-style forward ----
        try:
            out = self.model(pixel_values=images, return_dict=True)

            # Most common: last_hidden_state [B, 1+N, D] (CLS + patches)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                tokens = out.last_hidden_state
                if tokens.ndim != 3:
                    raise RuntimeError(f"Unexpected last_hidden_state shape: {tuple(tokens.shape)}")

                # drop CLS if present
                patch_tokens = tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens
                return patch_tokens

            # Some models might provide pooler_output [B, D]
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
                if pooled.ndim != 2:
                    raise RuntimeError(f"Unexpected pooler_output shape: {tuple(pooled.shape)}")
                return pooled.unsqueeze(1)  # [B,1,D]

        except TypeError:
            # signature mismatch; fall through
            pass

        # ---- Path B: custom extract_features API ----
        if hasattr(self.model, "extract_features"):
            out = self.model.extract_features(images)

            # dict outputs
            if isinstance(out, dict):
                for key in ["patch_tokens", "patchtokens", "x_norm_patchtokens", "tokens", "feat", "last_hidden_state"]:
                    if key in out:
                        pt = out[key]
                        if pt.ndim == 2:
                            return pt.unsqueeze(1)  # [B,1,D]
                        if pt.ndim == 3:
                            return pt[:, 1:, :] if pt.shape[1] > 1 else pt
                        return pt

            # tuple/list outputs
            if isinstance(out, (tuple, list)) and len(out) > 0:
                pt = out[0]
                if pt.ndim == 2:
                    return pt.unsqueeze(1)
                if pt.ndim == 3:
                    return pt[:, 1:, :] if pt.shape[1] > 1 else pt

        raise RuntimeError(
            "Could not extract patch tokens from RAD-JEPA. "
            "Model output didn't match HF pixel_values or extract_features patterns."
        )
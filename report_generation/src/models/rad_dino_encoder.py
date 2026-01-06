import torch
import torch.nn as nn


class FrozenRadDinoEncoder(nn.Module):
    """
    Frozen RAD-DINO encoder wrapper.

    Contract:
      input:  images [B,3,H,W] float in [0,1]
      output: patch_tokens [B,N,D] (CLS removed if present)
    """

    def __init__(self, rad_dino_model, image_processor=None):
        super().__init__()
        self.model = rad_dino_model
        self.processor = image_processor

        # freeze vision backbone
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # expose embedding dim for projector
        cfg = getattr(self.model, "config", None)
        self.embed_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
            or 768
        )

        # cache mean/std as BUFFERS (no attribute name collisions)
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
        """images float in [0,1], shape [B,3,H,W]"""
        mean = getattr(self, "_mean_buf", None)
        std = getattr(self, "_std_buf", None)
        if mean is None or std is None:
            return images
        return (images - mean.to(images.device)) / std.to(images.device)

    def _strip_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B,L,D] -> remove CLS if present"""
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected [B,L,D], got {tuple(tokens.shape)}")
        return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

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
                return self._strip_cls(tokens)
        except TypeError:
            pass

        # ---- Path B: custom extract_features API ----
        if hasattr(self.model, "extract_features"):
            out = self.model.extract_features(images)

            if isinstance(out, dict):
                for key in [
                    "patch_tokens",
                    "patchtokens",
                    "x_norm_patchtokens",
                    "tokens",
                    "feat",
                    "last_hidden_state",
                ]:
                    if key in out:
                        pt = out[key]
                        if pt.ndim == 2:
                            return pt.unsqueeze(1)
                        if pt.ndim == 3:
                            return self._strip_cls(pt)
                        raise RuntimeError(f"Unexpected token tensor shape for key={key}: {tuple(pt.shape)}")

            if isinstance(out, (tuple, list)) and len(out) > 0:
                pt = out[0]
                if pt.ndim == 2:
                    return pt.unsqueeze(1)
                if pt.ndim == 3:
                    return self._strip_cls(pt)
                raise RuntimeError(f"Unexpected token tensor shape from extract_features: {tuple(pt.shape)}")

        raise RuntimeError(
            "Could not extract patch tokens from RAD-DINO. "
            "Model output didn't match HF pixel_values or extract_features patterns."
        )
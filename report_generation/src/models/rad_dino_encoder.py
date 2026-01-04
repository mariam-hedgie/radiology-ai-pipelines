# src/models/rad_dino_encoder.py
import torch
import torch.nn as nn


class FrozenRadDinoEncoder(nn.Module):
    """
    Frozen RAD-DINO encoder that returns PATCH tokens: [B, N, 768].

    Works with:
    - HuggingFace ViT-style model: model(pixel_values=...) -> outputs.last_hidden_state
    - Or a custom model exposing extract_features(...)
    """

    def __init__(self, rad_dino_model, image_processor=None):
        """
        rad_dino_model: HF model OR your rad_dino model object
        image_processor: optional HF AutoImageProcessor (preferred),
                         used for mean/std normalization.
        """
        super().__init__()
        self.model = rad_dino_model
        self.processor = image_processor

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Cache mean/std for fast tensor normalization if processor provided
        if self.processor is not None:
            mean = getattr(self.processor, "image_mean", None)
            std = getattr(self.processor, "image_std", None)
            if mean is not None and std is not None:
                self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
                self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
            else:
                self.mean = None
                self.std = None
        else:
            self.mean = None
            self.std = None

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        images assumed float in [0,1], shape [B,3,H,W]
        """
        if self.mean is None or self.std is None:
            return images
        return (images - self.mean.to(images.device)) / self.std.to(images.device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W] float in [0,1]
        returns: patch_tokens [B,N,768]
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        # Normalize if processor exists (recommended)
        images = self._normalize(images)

        # ---- Path A: HuggingFace ViT-style forward ----
        # Most HF vision transformers accept pixel_values=...
        try:
            out = self.model(pixel_values=images, return_dict=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # out.last_hidden_state: [B, 1+N, 768] (CLS + patches)
                tokens = out.last_hidden_state
                patch_tokens = tokens[:, 1:, :]  # drop CLS
                return patch_tokens
        except TypeError:
            # model() signature doesn't match HF style; fall through
            pass

        # ---- Path B: custom extract_features API ----
        if hasattr(self.model, "extract_features"):
            out = self.model.extract_features(images)

            # Common possibilities:
            # - dict with patch tokens
            if isinstance(out, dict):
                for key in ["patch_tokens", "patchtokens", "x_norm_patchtokens", "tokens", "feat"]:
                    if key in out:
                        pt = out[key]
                        # If includes CLS, strip it
                        if pt.ndim == 3 and pt.shape[1] > 1 and pt.shape[2] == 768:
                            # ambiguous: could be [B,1+N,768] or [B,N,768]
                            # if it looks like CLS+patches, remove CLS
                            # safest heuristic: if token count is a perfect square +1 (ViT grid) => CLS present
                            # but keep it simple: remove first token if shape[1] looks like CLS+patches
                            return pt[:, 1:, :] if pt.shape[1] > 256 else pt
                        return pt

            # - tuple/list where first item is tokens
            if isinstance(out, (tuple, list)) and len(out) > 0:
                pt = out[0]
                if pt.ndim == 3:
                    return pt[:, 1:, :] if pt.shape[1] > 1 else pt

        raise RuntimeError(
            "Could not extract patch tokens. "
            "Your RAD-DINO model API didn't match HF pixel_values or extract_features patterns."
        )
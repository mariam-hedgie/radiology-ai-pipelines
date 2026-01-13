# src/backbones/ijepa_seg_backbone.py

import math
import torch
import torch.nn as nn
from transformers import AutoModel


class IJEPABackbone(nn.Module):
    """
    Frozen I-JEPA ViT-H/16 backbone that outputs intermediate feature maps for segmentation.

    Returns a list of feature maps at layers in out_indices:
        [B, C, h, w]  (C = hidden_size)

    Notes:
    - facebook/ijepa_vith16_1k expects 448x448 by default. We pass
      interpolate_pos_encoding=True so it can *usually* handle other sizes,
      but you should still resize to 448 for consistency.
    """

    def __init__(
        self,
        model_id: str = "facebook/ijepa_vith16_1k",
        out_indices=(7, 15, 23, 31),
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

        self.out_indices = set(out_indices)

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False

        # Useful metadata
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]
        returns: list of 4 feature maps [B, C, h, w]
        """

        out = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,  # key for non-448 sizes
        )

        # out.hidden_states is a tuple: (embeddings, layer1, layer2, ... layerL)
        # We want selected transformer layer outputs.
        hidden_states = out.hidden_states

        feats = []
        for layer_idx, hs in enumerate(hidden_states):
            # hidden_states includes embeddings at idx=0, then transformer blocks
            # so if you want "block i", you typically use i+1.
            if layer_idx in self.out_indices:
                feats.append(self._tokens_to_map(hs))

        return feats

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] or [B, 1+N, C]
        returns: [B, C, H, W]
        """

        B, N, C = tokens.shape

        # Some ViTs include CLS token (1+N). If so, drop it.
        h = int(math.sqrt(N))
        if h * h != N:
            # try dropping CLS
            if N > 1:
                tokens_ = tokens[:, 1:, :]
                N2 = tokens_.shape[1]
                h2 = int(math.sqrt(N2))
                if h2 * h2 == N2:
                    tokens = tokens_
                    N = N2
                    h = h2
                else:
                    raise ValueError(
                        f"Token count {N} is not a square (and N-1={N2} is also not square)."
                    )
            else:
                raise ValueError(f"Token count {N} is too small.")

        w = h
        return tokens.transpose(1, 2).reshape(B, C, h, w)
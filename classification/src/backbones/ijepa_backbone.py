# src/backbones/ijepa_backbone.py
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as e:
    raise ImportError(
        "Missing dependency: transformers. Install with: pip install transformers safetensors"
    ) from e


class HFImageBackbone(nn.Module):
    """
    Wraps a Hugging Face vision model and exposes:
      - embed_dim
      - forward_features(x) -> [B, C]
    so it plugs into your existing LinearProbeClassifier unchanged.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # most HF ViT-like configs use hidden_size
        self.embed_dim = getattr(model.config, "hidden_size", None)
        if self.embed_dim is None:
            raise AttributeError("HF model config missing hidden_size; can't infer embed_dim.")

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # HF models use pixel_values=
        out = self.model(pixel_values=x)
        # pool by mean over tokens (matches model card example)
        feats = out.last_hidden_state.mean(dim=1)  # [B, C]
        return feats


def build_ijepa_backbone(
    device: str | torch.device | None = None,
    model_name: str = "facebook/ijepa_vith16_1k",
) -> nn.Module:
    """
    Frozen I-JEPA ViT-H/16 backbone from Hugging Face.
    Returns features [B, C] using mean token pooling.
    """
    model = AutoModel.from_pretrained(model_name)

    # freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model = model.to(device)

    return HFImageBackbone(model)
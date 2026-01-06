# src/models/rad_jepa_encoder.py
import torch
import torch.nn as nn

try:
    from timm.models.vision_transformer import VisionTransformer
except ImportError as e:
    raise ImportError("timm is required. Install with: pip install timm") from e


class FrozenRadJepaEncoder(nn.Module):
    """
    Frozen LOCAL JEPA encoder wrapper (loads your .pth.tar checkpoint).

    Contract:
      input:  images [B,3,H,W] float in [0,1]
      output: patch_tokens [B,N,D]  (NO CLS token)
    """

    def __init__(
        self,
        ckpt_path: str,
        image_size: int = 224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__()

        if image_size != 224:
            raise ValueError("This local JEPA wrapper currently supports image_size=224 only.")

        # Build ViT-B/14 manually (matches your checkpoint shapes)
        self.model = VisionTransformer(
            img_size=224,
            patch_size=14,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            class_token=False,   # IMPORTANT: matches pos_embed length=256 (no CLS)
            global_pool="",      # keep token outputs
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "encoder" not in ckpt:
            raise RuntimeError(
                f"Expected a dict with key 'encoder' in {ckpt_path}. Got keys: {list(ckpt.keys())}"
            )

        sd = ckpt["encoder"]  # OrderedDict of tensors
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        # If these lists are huge, your architecture mismatch is real.
        if unexpected:
            print("[JEPA] Unexpected keys (sample):", unexpected[:10])
        if missing:
            print("[JEPA] Missing keys (sample):", missing[:10])

        # Freeze
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.embed_dim = 768

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self._mean.to(images.device)) / self._std.to(images.device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

        images = self._normalize(images)

        # forward_features returns tokens (since class_token=False, it’s [B,256,768])
        tokens = self.model.forward_features(images)

        if tokens.ndim != 3:
            raise RuntimeError(f"Expected [B,N,D], got {tuple(tokens.shape)}")

        return tokens
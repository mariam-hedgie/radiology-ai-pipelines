# src/backbones/rad_jepa_backbone.py
import torch
import torch.nn as nn
import timm

from src.backbones.vit_backbone import ViTBackbone


def _load_ckpt_flex(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common patterns:
    # 1) ckpt["state_dict"] is the actual model dict
    # 2) ckpt["encoder"] is the encoder weights (JEPA-style)
    # 3) ckpt["model"] or ckpt["student"] etc.
    state = ckpt.get("state_dict", ckpt)

    # If it's a JEPA training ckpt, the thing we want is usually state["encoder"]
    if isinstance(state, dict) and "encoder" in state and isinstance(state["encoder"], dict):
        state = state["encoder"]

    # Strip common prefixes
    new_state = {}
    for k, v in state.items():
        kk = k
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        new_state[kk] = v

    return new_state


def build_rad_jepa_backbone(
    jepa_ckpt: str,
    device: str | torch.device | None = None,
    img_size: int = 224,
) -> nn.Module:
    """
    Frozen RAD-JEPA-style ViT-B/16 backbone wrapper for classification.
    Returns CLS embedding [B, 768] via ViTBackbone wrapper.
    """
    # ViT-B/14 @ 224
    from timm.models.vision_transformer import VisionTransformer

    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=14,
        in_chans=3,
        num_classes=0,     # no head
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,     # <-- THIS matches your ckpt (3072)
        qkv_bias=True,
    )

    # Load JEPA weights
    state = _load_ckpt_flex(jepa_ckpt)

    # --- Fix JEPA pos_embed shape if checkpoint has no CLS position ---
    # timm ViT expects pos_embed with CLS token: [1, 1 + N, C]
    # your JEPA ckpt has [1, N, C]
    if "pos_embed" in state and hasattr(encoder, "pos_embed"):
        ckpt_pos = state["pos_embed"]
        model_pos = encoder.pos_embed

        if ckpt_pos.shape[1] == model_pos.shape[1] - 1:
            # prepend a CLS position (zeros) to match timm
            cls_pos = torch.zeros((ckpt_pos.shape[0], 1, ckpt_pos.shape[2]), dtype=ckpt_pos.dtype)
            state["pos_embed"] = torch.cat([cls_pos, ckpt_pos], dim=1)
            
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"[RadJEPA] loaded {jepa_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("[RadJEPA] unexpected (first 20):", unexpected[:20])
    if missing:
        print("[RadJEPA] missing (first 20):", missing[:20])

    if device is not None:
        encoder = encoder.to(device)

    # Freeze
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Wrap to standardize output to [B, 768]
    return ViTBackbone(encoder=encoder, embed_dim=768)
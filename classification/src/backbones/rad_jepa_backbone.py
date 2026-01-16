# src/backbones/rad_jepa_backbone.py
import torch
import torch.nn as nn
import timm

from src.backbones.vit_backbone import ViTBackbone

import torch.nn.functional as F


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

# added for 518 resizing
def _resize_pos_embed(ckpt_pos: torch.Tensor, model_pos: torch.Tensor) -> torch.Tensor:
    """
    Resize checkpoint pos_embed to match model pos_embed using bicubic interpolation.

    - model_pos (timm ViT): [1, 1 + N, C] (includes CLS)
    - ckpt_pos (JEPA often): [1, N, C] (no CLS) OR [1, 1+N, C] (has CLS)
    """
    C = model_pos.shape[-1]
    model_tokens = model_pos.shape[1]          # 1 + N_new
    model_patches = model_tokens - 1           # N_new

    # If already matches, no-op
    if ckpt_pos.shape[1] == model_tokens:
        return ckpt_pos
    
    # If checkpoint matches patch count but has no CLS, just prepend CLS (no interpolation)
    if ckpt_pos.shape[1] == model_patches:
        cls_pos = torch.zeros((1, 1, C), dtype=ckpt_pos.dtype)
        return torch.cat([cls_pos, ckpt_pos], dim=1)

    # Decide whether checkpoint includes CLS by checking which length forms a square grid
    n_tok = ckpt_pos.shape[1]

    def is_square(n: int) -> bool:
        s = int(n ** 0.5)
        return s * s == n

    if is_square(n_tok):
        # No CLS token, it's pure patch grid (e.g. 256 = 16x16)
        ckpt_cls = None
        ckpt_patch = ckpt_pos                  # [1, N_old, C]
    elif is_square(n_tok - 1):
        # Has CLS token at the front
        ckpt_cls = ckpt_pos[:, :1, :]          # [1, 1, C]
        ckpt_patch = ckpt_pos[:, 1:, :]        # [1, N_old, C]
    else:
        raise AssertionError(
            f"Can't infer CLS token for ckpt pos_embed with {n_tok} tokens "
            f"(neither {n_tok} nor {n_tok-1} is a perfect square). "
            "This checkpoint may use extra tokens (dist_token, etc.)."
        )

    old_n = ckpt_patch.shape[1]
    old_size = int(old_n ** 0.5)
    new_size = int(model_patches ** 0.5)

    assert old_size * old_size == old_n, f"ckpt pos_embed patches not square: {old_n}"
    assert new_size * new_size == model_patches, f"model pos_embed patches not square: {model_patches}"

    # [1, N, C] -> [1, C, H, W]
    ckpt_patch = ckpt_patch.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)

    # Resize grid
    ckpt_patch = torch.nn.functional.interpolate(
        ckpt_patch, size=(new_size, new_size), mode="bicubic", align_corners=False
    )

    # [1, C, H, W] -> [1, N, C]
    ckpt_patch = ckpt_patch.permute(0, 2, 3, 1).reshape(1, new_size * new_size, C)

    # Add CLS back
    if ckpt_cls is None:
        ckpt_cls = torch.zeros((1, 1, C), dtype=ckpt_patch.dtype)
    return torch.cat([ckpt_cls, ckpt_patch], dim=1)  # [1, 1+N_new, C]


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
        mlp_ratio=4.0,     
        qkv_bias=True,
    )

    # Load JEPA weights
    state = _load_ckpt_flex(jepa_ckpt)

    # --- Fix JEPA pos_embed shape if checkpoint has no CLS position ---
    # timm ViT expects pos_embed with CLS token: [1, 1 + N, C]
    # your JEPA ckpt has [1, N, C]

    # for 518 resizing 
    # if "pos_embed" in state and hasattr(encoder, "pos_embed"):
    #     ckpt_pos = state["pos_embed"]
    #     model_pos = encoder.pos_embed

    #     if ckpt_pos.shape[1] == model_pos.shape[1] - 1:
    #         # prepend a CLS position (zeros) to match timm
    #         cls_pos = torch.zeros((ckpt_pos.shape[0], 1, ckpt_pos.shape[2]), dtype=ckpt_pos.dtype)
    #         state["pos_embed"] = torch.cat([cls_pos, ckpt_pos], dim=1)

    # added for 518 resizing
    # --- Fix/resize pos_embed to match the model's img_size ---
    # Needed when you change img_size (e.g., 224 -> 518)
    if "pos_embed" in state and hasattr(encoder, "pos_embed"):
        state["pos_embed"] = _resize_pos_embed(state["pos_embed"], encoder.pos_embed)
            
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
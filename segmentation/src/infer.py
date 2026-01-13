# src/infer.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.datasets.lung_dataset import LungSegDataset
from src.model import UPerNetSegModel


# ----------------- utils -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [3,H,W] float (usually normalized) -> uint8 [H,W,3] for visualization.
    We rescale per-image to [0,1] defensively so it looks sane regardless of normalization.
    """
    x = img_tensor.detach().cpu().float()

    # robust min/max rescale for display
    x = x - x.min()
    x = x / (x.max().clamp(min=1e-6))

    x = (x * 255.0).clamp(0, 255).byte()
    return x.permute(1, 2, 0).numpy()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    mask: [H,W] values {0,1}
    return colored mask [H,W,3] for visualization
    """
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)  # lung = green
    return color


def overlay(img: np.ndarray, colored_mask: np.ndarray, alpha=0.35) -> np.ndarray:
    """alpha blend mask on image"""
    return (img.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha).astype(np.uint8)


# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser()

    # data / io
    parser.add_argument("--data_root", type=str, default="data/lung_seg")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--save_max", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=2)

    # model
    parser.add_argument("--ckpt", type=str, required=True, help="Segmentation model checkpoint (.pth)")
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Required if backbone=jepa")
    # if your UPerNetSegModel needs anything extra for iJEPA, add args here later (e.g. --ijepa_model_id)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # --- paths
    data_root = Path(args.data_root)
    img_dir = data_root / args.split / "images"
    mask_dir = data_root / args.split / "masks"
    assert img_dir.exists(), f"Missing: {img_dir}"
    assert mask_dir.exists(), f"Missing: {mask_dir}"

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    out_dir = Path(args.out_root) / args.backbone / args.split / ckpt_path.stem
    ensure_dir(out_dir)

    # --- data
    ds = LungSegDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- model
    model_kwargs = dict(
        num_classes=args.num_classes,
        backbone=args.backbone,
    )
    if args.backbone == "jepa":
        if args.jepa_ckpt is None:
            raise ValueError("For --backbone jepa you must pass --jepa_ckpt /path/to/best_jepa_weights.pth.tar")
        model_kwargs["jepa_ckpt"] = args.jepa_ckpt

    # NOTE: iJEPA should be handled internally by UPerNetSegModel(backbone="ijepa")
    model = UPerNetSegModel(**model_kwargs).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded segmentation ckpt: {ckpt_path}")
    print(f"Backbone: {args.backbone}")
    if args.backbone == "jepa":
        print(f"JEPA encoder ckpt: {args.jepa_ckpt}")
    print(f"Saving visualizations to: {out_dir}")

    saved = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)   # [B,3,H,W]
            masks = masks.to(device, non_blocking=True) # [B,H,W] (or [B,1,H,W] depending on dataset)

            # normalize mask shape to [B,H,W]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

            logits = model(imgs)                        # [B,C,H,W]
            pred = torch.argmax(logits, dim=1)          # [B,H,W]

            bs = imgs.size(0)
            for b in range(bs):
                if saved >= args.save_max:
                    break

                img_np = to_uint8(imgs[b])
                gt_np = masks[b].detach().cpu().numpy().astype(np.uint8)
                pred_np = pred[b].detach().cpu().numpy().astype(np.uint8)

                gt_overlay = overlay(img_np, mask_to_color(gt_np), alpha=0.35)
                pred_overlay = overlay(img_np, mask_to_color(pred_np), alpha=0.35)

                # save
                Image.fromarray(img_np).save(out_dir / f"{saved:04d}_img.png")
                Image.fromarray(gt_overlay).save(out_dir / f"{saved:04d}_gt_overlay.png")
                Image.fromarray(pred_overlay).save(out_dir / f"{saved:04d}_pred_overlay.png")
                Image.fromarray((pred_np * 255).astype(np.uint8)).save(out_dir / f"{saved:04d}_pred_mask.png")

                saved += 1

            if saved >= args.save_max:
                break

    print(f"Done. Saved {saved} samples to: {out_dir}")


if __name__ == "__main__":
    main()
# src/infer.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.datasets.lung_dataset import LungSegDataset
from src.datasets.vindr_rib_dataset import VinDrRibSegDataset
from src.model import UPerNetSegModel


# ----------------- utils -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [3,H,W] float (maybe normalized) -> uint8 [H,W,3] for visualization.
    Rescale per-image to [0,1] so it looks ok regardless of normalization.
    """
    x = img_tensor.detach().cpu().float()
    x = x - x.min()
    x = x / (x.max().clamp(min=1e-6))
    x = (x * 255.0).clamp(0, 255).byte()
    return x.permute(1, 2, 0).numpy()


def mask_to_color(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    mask: [H,W] int labels in [0..num_classes-1]
    returns RGB visualization. Class 0 = black background.
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    if num_classes <= 2:
        # binary: class 1 = green
        color[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)
        return color

    # multiclass: deterministic palette (no randomness)
    palette = np.array([
        [0, 0, 0],       # 0 background
        [0, 255, 0],     # 1
        [255, 0, 0],     # 2
        [0, 0, 255],     # 3
        [255, 255, 0],   # 4
        [255, 0, 255],   # 5
        [0, 255, 255],   # 6
        [255, 127, 0],   # 7
        [127, 0, 255],   # 8
        [0, 127, 255],   # 9
    ], dtype=np.uint8)

    for c in range(1, num_classes):
        color[mask == c] = palette[c % len(palette)]

    return color


def overlay(img: np.ndarray, colored_mask: np.ndarray, alpha=0.35) -> np.ndarray:
    return (img.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha).astype(np.uint8)


# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser()

    # dataset choice
    parser.add_argument("--dataset", type=str, required=True, choices=["lung", "vindr_rib"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    # lung args
    parser.add_argument("--data_root", type=str, default="data/lung_seg")

    # vindr-rib args
    parser.add_argument("--ann_json", type=str, default=None, help="Path to Vindr_RibCXR_*_mask.json")
    parser.add_argument("--image_root", type=str, default=None, help="Root containing Data/train/img etc.")
    parser.add_argument("--rib_mode", type=str, default="multiclass", choices=["multiclass", "binary"])

    # io
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--save_max", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=2)

    # model / backbone
    parser.add_argument("--ckpt", type=str, required=True, help="Segmentation model checkpoint (.pth)")
    parser.add_argument("--num_classes", type=int, default=None, help="Override num_classes; otherwise infer from dataset")

    parser.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Required if backbone=jepa")
    parser.add_argument("--ijepa_model_id", type=str, default="facebook/ijepa_vith16_1k")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    # ----- IMPORTANT: match input size to backbone -----
    image_size = 448 if args.backbone == "ijepa" else 224

    # ----- dataset -----
    if args.dataset == "lung":
        data_root = Path(args.data_root)
        img_dir = data_root / args.split / "images"
        mask_dir = data_root / args.split / "masks"
        assert img_dir.exists(), f"Missing: {img_dir}"
        assert mask_dir.exists(), f"Missing: {mask_dir}"

        ds = LungSegDataset(str(img_dir), str(mask_dir), image_size=image_size)
        inferred_num_classes = 2

    else:  # vindr_rib
        if args.ann_json is None or args.image_root is None:
            raise ValueError("For --dataset vindr_rib you must pass --ann_json and --image_root")

        ds = VinDrRibSegDataset(
            ann_json=args.ann_json,
            image_root=args.image_root,
            split=args.split,
            image_size=image_size,
            mode=args.rib_mode,
        )
        inferred_num_classes = ds.num_classes

    num_classes = args.num_classes if args.num_classes is not None else inferred_num_classes

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- output dir -----
    out_dir = Path(args.out_root) / args.dataset / args.backbone / args.split / ckpt_path.stem
    ensure_dir(out_dir)

    # ----- model -----
    model_kwargs = dict(
        num_classes=num_classes,
        backbone=args.backbone,
        ijepa_model_id=args.ijepa_model_id,
    )
    if args.backbone == "jepa":
        if args.jepa_ckpt is None:
            raise ValueError("For --backbone jepa you must pass --jepa_ckpt /path/to/weights.pth.tar")
        model_kwargs["jepa_ckpt"] = args.jepa_ckpt

    model = UPerNetSegModel(**model_kwargs).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded segmentation ckpt: {ckpt_path}")
    print(f"Dataset: {args.dataset} | Split: {args.split}")
    print(f"Backbone: {args.backbone} | image_size={image_size} | num_classes={num_classes}")
    if args.backbone == "jepa":
        print(f"JEPA encoder ckpt: {args.jepa_ckpt}")
    print(f"Saving visualizations to: {out_dir}")

    saved = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)    # [B,3,H,W]
            masks = masks.to(device, non_blocking=True)  # [B,H,W] or [B,1,H,W]

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

            logits = model(imgs)              # [B,C,H,W]
            pred = torch.argmax(logits, dim=1)  # [B,H,W]

            bs = imgs.size(0)
            for b in range(bs):
                if saved >= args.save_max:
                    break

                img_np = to_uint8(imgs[b])
                gt_np = masks[b].detach().cpu().numpy().astype(np.int32)
                pred_np = pred[b].detach().cpu().numpy().astype(np.int32)

                gt_overlay = overlay(img_np, mask_to_color(gt_np, num_classes), alpha=0.35)
                pred_overlay = overlay(img_np, mask_to_color(pred_np, num_classes), alpha=0.35)

                Image.fromarray(img_np).save(out_dir / f"{saved:04d}_img.png")
                Image.fromarray(gt_overlay).save(out_dir / f"{saved:04d}_gt_overlay.png")
                Image.fromarray(pred_overlay).save(out_dir / f"{saved:04d}_pred_overlay.png")

                # raw mask visualization (scaled)
                if num_classes <= 2:
                    vis = (pred_np * 255).astype(np.uint8)
                else:
                    vis = (pred_np.astype(np.float32) / max(1, num_classes - 1) * 255.0).astype(np.uint8)
                Image.fromarray(vis).save(out_dir / f"{saved:04d}_pred_mask.png")

                saved += 1

            if saved >= args.save_max:
                break

    print(f"Done. Saved {saved} samples to: {out_dir}")


if __name__ == "__main__":
    main()
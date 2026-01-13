import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lung_dataset import LungSegDataset
from src.model import UPerNetSegModel


@torch.no_grad()
def dice_iou_binary(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: [H,W] with values {0,1}
    returns: (dice, iou)
    """
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    pred_sum = pred.sum().float()
    gt_sum = gt.sum().float()

    dice = (2 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice.item(), iou.item()


@torch.no_grad()
def dice_iou_multiclass(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, ignore_index: int | None = 0):
    """
    pred, gt: [H,W] with values in [0..K-1]
    Computes per-class Dice/IoU, then returns:
      - macro (average over classes, excluding ignore_index if provided)
      - per_class arrays

    NOTE: For lung-zone style tasks, you'll want ignore_index=None or ignore_index=0 depending on how you encode background.
    """
    dice_per = np.full((num_classes,), np.nan, dtype=np.float64)
    iou_per = np.full((num_classes,), np.nan, dtype=np.float64)

    classes = list(range(num_classes))
    if ignore_index is not None and ignore_index in classes:
        classes.remove(ignore_index)

    for c in classes:
        p = (pred == c)
        g = (gt == c)

        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        p_sum = p.sum().item()
        g_sum = g.sum().item()

        dice = (2 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dice_per[c] = dice
        iou_per[c] = iou

    macro_dice = float(np.nanmean(dice_per))
    macro_iou = float(np.nanmean(iou_per))
    return macro_dice, macro_iou, dice_per, iou_per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pth")
    parser.add_argument("--data_root", type=str, default="data/lung_seg")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--backbone", type=str, default="jepa", choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="required if backbone=jepa")
    parser.add_argument("--ignore_index", type=int, default=0, help="background class to ignore for multiclass macro; set -1 to disable")

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = Path(args.data_root) / args.split / "images"
    mask_dir = Path(args.data_root) / args.split / "masks"
    assert img_dir.exists() and mask_dir.exists(), "Missing split folders."

    ds = LungSegDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # model
    model = UPerNetSegModel(
        num_classes=args.num_classes,
        backbone=args.backbone,
        jepa_ckpt=args.jepa_ckpt
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # collect per-image metrics (this is what you compute mean/std over)
    dice_list = []
    iou_list = []

    # for multiclass: collect per-class too
    per_class_dice = []
    per_class_iou = []

    ignore_index = None if args.ignore_index == -1 else args.ignore_index

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Evaluating {args.split}"):
            imgs = imgs.to(device)     # [1,3,H,W]
            masks = masks.to(device)   # [1,H,W] long

            logits = model(imgs)                 # [1,C,H,W]
            pred = torch.argmax(logits, dim=1)   # [1,H,W]

            p = pred[0]
            g = masks[0]

            if args.num_classes == 2:
                # binary: assume foreground=1
                d, j = dice_iou_binary(p, g)
                dice_list.append(d)
                iou_list.append(j)
            else:
                # multiclass: macro over classes (optionally ignoring background)
                d_macro, j_macro, d_per, j_per = dice_iou_multiclass(
                    p, g, num_classes=args.num_classes, ignore_index=ignore_index
                )
                dice_list.append(d_macro)
                iou_list.append(j_macro)
                per_class_dice.append(d_per)
                per_class_iou.append(j_per)

    dice_arr = np.array(dice_list, dtype=np.float64)
    iou_arr = np.array(iou_list, dtype=np.float64)

    dice_mean = float(dice_arr.mean())
    dice_std = float(dice_arr.std(ddof=0))  # population std to match most ML tables
    iou_mean = float(iou_arr.mean())
    iou_std = float(iou_arr.std(ddof=0))

    print(f"\nCheckpoint: {args.ckpt}")
    print(f"Split: {args.split} | N={len(ds)}")
    print(f"Backbone: {args.backbone} | num_classes={args.num_classes}")

    # paper-style: mean (std)
    print(f"\nDice: {dice_mean*100:.1f} ({dice_std*100:.1f})")
    print(f"IoU : {iou_mean*100:.1f} ({iou_std*100:.1f})")

    # optional: per-class breakdown (useful for lung zones etc.)
    if args.num_classes > 2 and len(per_class_dice) > 0:
        dmat = np.stack(per_class_dice, axis=0)  # [N,K]
        imat = np.stack(per_class_iou, axis=0)   # [N,K]

        print("\nPer-class Dice mean (std):")
        for c in range(args.num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            m = np.nanmean(dmat[:, c])
            s = np.nanstd(dmat[:, c], ddof=0)
            print(f"  class {c}: {m*100:.1f} ({s*100:.1f})")

        print("\nPer-class IoU mean (std):")
        for c in range(args.num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            m = np.nanmean(imat[:, c])
            s = np.nanstd(imat[:, c], ddof=0)
            print(f"  class {c}: {m*100:.1f} ({s*100:.1f})")


if __name__ == "__main__":
    main()
# scripts/eval_test_dice_iou.py
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lung_dataset import LungSegDataset
from src.datasets.vindr_rib_dataset import VinDrRibSegDataset
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
    return float(dice.item()), float(iou.item())


@torch.no_grad()
def dice_iou_multiclass(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = 0,
):
    """
    pred, gt: [H,W] with values in [0..K-1]
    Returns:
      macro_dice, macro_iou, dice_per[K], iou_per[K]
    """
    dice_per = np.full((num_classes,), np.nan, dtype=np.float64)
    iou_per = np.full((num_classes,), np.nan, dtype=np.float64)

    classes = list(range(num_classes))
    if ignore_index is not None and ignore_index in classes:
        classes.remove(ignore_index)

    for c in classes:
        p = (pred == c)
        g = (gt == c)

        inter = int((p & g).sum().item())
        union = int((p | g).sum().item())
        p_sum = int(p.sum().item())
        g_sum = int(g.sum().item())

        dice = (2 * inter + 1e-6) / (p_sum + g_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dice_per[c] = dice
        iou_per[c] = iou

    macro_dice = float(np.nanmean(dice_per))
    macro_iou = float(np.nanmean(iou_per))
    return macro_dice, macro_iou, dice_per, iou_per


def main():
    parser = argparse.ArgumentParser()

    # ---------------- dataset ----------------
    parser.add_argument("--dataset", type=str, required=True, choices=["lung", "vindr_rib"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    # lung dataset args
    parser.add_argument("--data_root", type=str, default="data/lung_seg")

    # vindr_rib dataset args
    parser.add_argument("--ann_json", type=str, default=None, help="Path to Vindr_RibCXR_*_mask.json")
    parser.add_argument("--image_root", type=str, default=None, help="Root containing Data/train/img etc.")
    parser.add_argument("--rib_mode", type=str, default="multiclass", choices=["multiclass", "binary"])

    # ---------------- model ----------------
    parser.add_argument("--ckpt", type=str, required=True, help="path to segmentation checkpoint (.pth)")
    parser.add_argument("--num_classes", type=int, default=None, help="Override; else inferred from dataset")

    parser.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="required if backbone=jepa")
    parser.add_argument("--ijepa_model_id", type=str, default="facebook/ijepa_vith16_1k")

    # metrics
    parser.add_argument("--ignore_index", type=int, default=0,
                        help="class to ignore for multiclass macro. Use -1 to disable ignore.")

    # loader/device
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"

    # ---- IMPORTANT: match input size to backbone ----
    image_size = 448 if args.backbone == "ijepa" else 224

    # ---- build dataset ----
    if args.dataset == "lung":
        img_dir = Path(args.data_root) / args.split / "images"
        mask_dir = Path(args.data_root) / args.split / "masks"
        assert img_dir.exists() and mask_dir.exists(), "Missing split folders."

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

    # ---- build model ----
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

    ignore_index = None if args.ignore_index == -1 else args.ignore_index

    # ---- collect per-image metrics ----
    dice_list = []
    iou_list = []

    # multiclass: also keep per-class per-image arrays
    per_class_dice = []
    per_class_iou = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Eval {args.dataset}:{args.split}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # normalize mask shape to [B,H,W]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

            logits = model(imgs)               # [B,C,H,W]
            pred = torch.argmax(logits, dim=1) # [B,H,W]

            bs = imgs.size(0)
            for b in range(bs):
                p = pred[b]
                g = masks[b]

                if num_classes == 2:
                    d, j = dice_iou_binary(p, g)
                    dice_list.append(d)
                    iou_list.append(j)
                else:
                    d_macro, j_macro, d_per, j_per = dice_iou_multiclass(
                        p, g, num_classes=num_classes, ignore_index=ignore_index
                    )
                    dice_list.append(d_macro)
                    iou_list.append(j_macro)
                    per_class_dice.append(d_per)
                    per_class_iou.append(j_per)

    dice_arr = np.array(dice_list, dtype=np.float64)
    iou_arr = np.array(iou_list, dtype=np.float64)

    # mean/std OVER IMAGES (this matches “mean (std)” style tables)
    dice_mean = float(np.nanmean(dice_arr))
    dice_std = float(np.nanstd(dice_arr, ddof=0))
    iou_mean = float(np.nanmean(iou_arr))
    iou_std = float(np.nanstd(iou_arr, ddof=0))

    print("\n================ RESULTS ================")
    print(f"Dataset: {args.dataset} | Split: {args.split} | N={len(ds)}")
    print(f"Backbone: {args.backbone} | image_size={image_size} | num_classes={num_classes}")
    print(f"Checkpoint: {ckpt_path}")

    # paper-style: mean (std), often shown as percent
    print(f"\nDice: {dice_mean*100:.1f} ({dice_std*100:.1f})")
    print(f"IoU : {iou_mean*100:.1f} ({iou_std*100:.1f})")

    if num_classes > 2 and len(per_class_dice) > 0:
        dmat = np.stack(per_class_dice, axis=0)  # [N,K]
        imat = np.stack(per_class_iou, axis=0)   # [N,K]

        print("\nPer-class Dice mean (std):")
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            m = float(np.nanmean(dmat[:, c]))
            s = float(np.nanstd(dmat[:, c], ddof=0))
            print(f"  class {c:02d}: {m*100:.1f} ({s*100:.1f})")

        print("\nPer-class IoU mean (std):")
        for c in range(num_classes):
            if ignore_index is not None and c == ignore_index:
                continue
            m = float(np.nanmean(imat[:, c]))
            s = float(np.nanstd(imat[:, c], ddof=0))
            print(f"  class {c:02d}: {m*100:.1f} ({s*100:.1f})")


if __name__ == "__main__":
    main()
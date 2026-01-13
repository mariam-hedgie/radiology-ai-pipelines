import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lung_dataset import LungSegDataset
#from src.model import RadDinoUPerNet
from src.model import UPerNetSegModel


def dice_iou(pred, gt):
    """
    pred, gt: torch tensors [H,W] with values {0,1}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoints/epoch_10.pth")
    parser.add_argument("--data_root", type=str, default="data/lung_seg")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    
    parser.add_argument("--backbone", type=str, default="jepa", choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default="/data1/mariam/best_jepa_weights.pth.tar")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = Path(args.data_root) / args.split / "images"
    mask_dir = Path(args.data_root) / args.split / "masks"
    assert img_dir.exists() and mask_dir.exists(), "Missing test split folders."

    ds = LungSegDataset(str(img_dir), str(mask_dir))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    #model = RadDinoUPerNet(num_classes=args.num_classes).to(device)
    model = UPerNetSegModel(
        num_classes=args.num_classes,
        backbone=args.backbone,
        jepa_ckpt=args.jepa_ckpt if args.backbone == "jepa" else None,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"Evaluating {args.split}"):
            imgs = imgs.to(device)
            masks = masks.to(device)  # [1,H,W] values {0,1}

            logits = model(imgs)             # [1,C,H,W]
            pred = torch.argmax(logits, dim=1)  # [1,H,W]

            d, j = dice_iou(pred[0], masks[0])
            dices.append(d)
            ious.append(j)

    print(f"Checkpoint: {args.ckpt}")
    print(f"{args.split} Mean Dice: {np.mean(dices):.4f} | Median Dice: {np.median(dices):.4f}")
    print(f"{args.split} Mean IoU : {np.mean(ious):.4f} | Median IoU : {np.median(ious):.4f}")
    print(f"Min Dice: {np.min(dices):.4f} | Max Dice: {np.max(dices):.4f}")


if __name__ == "__main__":
    main()
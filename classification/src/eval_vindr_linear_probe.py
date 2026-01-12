# scripts/eval_vindr_linear_probe.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import wandb

from torchvision import transforms
from PIL import Image

from torchmetrics.classification import MultilabelAveragePrecision

from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone
from src.backbones.rad_jepa_backbone import build_rad_jepa_backbone


class VinDrCXRImageLabels(Dataset):
    """
    VinDr-CXR image-level MULTI-LABEL dataset.
    Aggregates multiple radiologists per image_id using max() (OR rule).
    """
    def __init__(self, split_dir: str, csv_path: str, transform=None):
        self.split_dir = Path(split_dir)
        df = pd.read_csv(csv_path)
        self.transform = transform

        self.label_cols = list(df.columns[2:])
        self.num_classes = len(self.label_cols)

        grouped = df.groupby("image_id")[self.label_cols].max().reset_index()
        self.image_ids = grouped["image_id"].tolist()
        self.labels = grouped[self.label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.image_ids)

    def _find_dicom(self, image_id: str) -> Path:
        for ext in [".dicom", ".dcm", ""]:
            p = self.split_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        hits = list(self.split_dir.glob(f"{image_id}.*"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Missing DICOM for image_id={image_id} under {self.split_dir}")

    def _read_dicom_to_uint8(self, dicom_path: Path) -> np.ndarray:
        dcm = pydicom.dcmread(str(dicom_path))
        arr = dcm.pixel_array.astype(np.float32)

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        # robust contrast scaling
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / max(1e-6, (hi - lo))
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        y = self.labels[idx]  # float32 [K]
        dicom_path = self._find_dicom(image_id)

        img_u8 = self._read_dicom_to_uint8(dicom_path)
        img = Image.fromarray(img_u8).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, torch.from_numpy(y)


def load_head_only(model: LinearProbeClassifier, ckpt_path: str):
    """
    Loads ONLY classifier weights from a checkpoint containing either:
      - full model.state_dict()
      - {"model": state_dict} or {"state_dict": state_dict}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # normalize keys: remove common prefixes
    norm = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        norm[k] = v

    # extract classifier only
    head_state = {}
    for k, v in norm.items():
        if k.startswith("classifier."):
            head_state[k[len("classifier."):]] = v

    if not head_state:
        raise RuntimeError(
            f"No classifier.* keys found in {ckpt_path}. "
            f"First 30 keys: {list(norm.keys())[:30]}"
        )

    model.classifier.load_state_dict(head_state, strict=True)
    print(f"Loaded head from {ckpt_path}: {list(head_state.keys())}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backbone", type=str, default="dino", choices=["dino", "jepa"])
    ap.add_argument("--jepa_ckpt", type=str, default=None)

    ap.add_argument("--dicom_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--per_class", action="store_true")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="vindr-linear-probe")
    ap.add_argument("--wandb_name", type=str, default="eval")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds = VinDrCXRImageLabels(args.dicom_dir, args.labels_csv, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    num_classes = ds.num_classes
    class_names = ds.label_cols

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "ckpt": args.ckpt,
                "num_classes": num_classes,
                "backbone": args.backbone,
                "dicom_dir": args.dicom_dir,
                "labels_csv": args.labels_csv,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
            }
        )

    # backbone
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)
    else:
        if args.jepa_ckpt is None:
            raise ValueError("Need --jepa_ckpt when backbone=jepa")
        backbone = build_rad_jepa_backbone(jepa_ckpt=args.jepa_ckpt, device=device)

    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    load_head_only(model, args.ckpt)
    model.eval()

    # multi-label loss
    criterion = nn.BCEWithLogitsLoss()

    # metrics
    auprc_macro = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(device)
    auprc_per = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)

    loss_sum = 0.0
    n = 0

    auprc_macro.reset()
    if args.per_class:
        auprc_per.reset()

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()  # BCE expects float targets

            logits = model(imgs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            loss_sum += loss.item() * imgs.size(0)
            n += imgs.size(0)

            auprc_macro.update(probs, targets.int())
            if args.per_class:
                auprc_per.update(probs, targets.int())

    avg_loss = loss_sum / max(1, n)
    macro = auprc_macro.compute().item()

    print(f"Loss: {avg_loss:.4f} | AUPRC(macro): {macro:.4f}")

    log_out = {"loss": avg_loss, "auprc_macro": macro}

    if args.per_class:
        per_ap = auprc_per.compute().detach().cpu().numpy()
        auprc_std = float(np.std(per_ap))
        print(f"AUPRC std across classes: {auprc_std:.4f}")

        top_idx = np.argsort(-per_ap)[:10]
        bottom_idx = np.argsort(per_ap)[:10]

        print("\nTop 10 classes by AUPRC:")
        for i in top_idx:
            print(f"  {class_names[i]}: {per_ap[i]:.4f}")

        print("\nBottom 10 classes by AUPRC:")
        for i in bottom_idx:
            print(f"  {class_names[i]}: {per_ap[i]:.4f}")

        log_out["auprc_std"] = auprc_std

        # optional: log per-class AUPRCs (careful: many keys)
        # for i, name in enumerate(class_names):
        #     log_out[f"auprc/{name}"] = float(per_ap[i])

    if args.wandb:
        wandb.log(log_out)
        wandb.finish()


if __name__ == "__main__":
    main()
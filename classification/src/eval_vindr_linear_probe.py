# scripts/eval_vindr_linear_probe.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb

from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelF1Score, MultilabelConfusionMatrix

from src.datasets.vindr_dataset import VinDrCXRImageLabels
from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone
from src.backbones.rad_jepa_backbone import build_rad_jepa_backbone
from src.backbones.ijepa_backbone import build_ijepa_backbone

def load_head_only(model: LinearProbeClassifier, ckpt_path: str):
    """
    Loads ONLY classifier weights from a checkpoint that contains either:
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

    # normalize prefixes
    norm = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        norm[k] = v

    # extract classifier.* only
    head_state = {}
    for k, v in norm.items():
        if k.startswith("classifier."):
            head_state[k[len("classifier."):]] = v

    if not head_state:
        raise RuntimeError(
            f"No classifier.* keys found in {ckpt_path}. "
            f"Keys (first 30): {list(norm.keys())[:30]}"
        )

    model.classifier.load_state_dict(head_state, strict=True)


@torch.no_grad()
def eval_one_ckpt(model, loader, device, num_classes: int, f1_thresh: float, keep_idx=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    auprc_macro = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(device)
    auprc_per = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)

    f1_macro = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=f1_thresh).to(device)
    f1_per   = MultilabelF1Score(num_labels=num_classes, average=None, threshold=f1_thresh).to(device)
    cm_per = MultilabelConfusionMatrix(num_labels=num_classes, threshold=f1_thresh).to(device)

    loss_sum = 0.0
    n = 0

    auprc_macro.reset()
    auprc_per.reset()

    f1_macro.reset()
    f1_per.reset()

    cm_per.reset()

    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        if keep_idx is not None:
            targets = targets[:, keep_idx]

        logits = model(imgs)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)

        auprc_macro.update(probs, targets.int())
        auprc_per.update(probs, targets.int())

        f1_macro.update(probs, targets.int())
        f1_per.update(probs, targets.int())

        cm_per.update(probs, targets.int())

    avg_loss = loss_sum / max(1, n)
    macro = float(auprc_macro.compute().item())
    per_ap = auprc_per.compute().detach().cpu().numpy().astype(np.float64)  # [K]

    f1m = float(f1_macro.compute().item())
    per_f1 = f1_per.compute().detach().cpu().numpy().astype(np.float64)

    per_cm = cm_per.compute().detach().cpu().numpy().astype(np.int64)  # shape [K, 2, 2]

    return avg_loss, macro, per_ap, f1m, per_f1, per_cm


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    ap.add_argument("--jepa_ckpt", type=str, default=None, help="Required when backbone=jepa")

    ap.add_argument("--dicom_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)

    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--f1_thresh", type=float, default=0.5, help="threshold for multilabel F1")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="vindr-linear-probe")
    ap.add_argument("--wandb_name", type=str, default="eval")

    ap.add_argument("--keep_7", action="store_true", help="Evaluate only the 7 target labels")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # change resolution as needed
    if args.backbone == "ijepa":
        image_size = 448
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif args.backbone == "jepa": # changing image size to 518 for jepa
        image_size = 518
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        image_size = args.image_size
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # Use the SAME dataset class as training
    ds = VinDrCXRImageLabels(
        dicom_dir=args.dicom_dir,
        labels_csv=args.labels_csv,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = ds.num_classes
    class_names = ds.label_cols

    keep_idx = None
    if args.keep_7:
        keep_labels = [
            "Lung Opacity",
            "Cardiomegaly",
            "Pleural thickening",
            "Aortic enlargement",
            "Pulmonary fibrosis",
            "Tuberculosis",
            "Pleural effusion",
        ]
        missing = [l for l in keep_labels if l not in class_names]
        if missing:
            raise ValueError(f"Missing labels in CSV header: {missing}")
        
        keep_idx = [class_names.index(l) for l in keep_labels]
        class_names = keep_labels
        num_classes = len(keep_idx)

        print("Evaluating only labels:", keep_labels)
        print("Indices in CSV", keep_idx)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "backbone": args.backbone,
                "dicom_dir": args.dicom_dir,
                "labels_csv": args.labels_csv,
                "image_size": image_size,
                "batch_size": args.batch_size,
                "ckpt": args.ckpt,
            },
        )

    # ---- build backbone ----
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)
    elif args.backbone == "jepa":
        if args.jepa_ckpt is None:
            raise ValueError("Need --jepa_ckpt when backbone=jepa")
        backbone = build_rad_jepa_backbone(jepa_ckpt=args.jepa_ckpt, device=device)
    else:
        backbone = build_ijepa_backbone(device=device)

    # model
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ---- load head and eval ----
    ckpt_path = str(Path(args.ckpt))
    load_head_only(model, ckpt_path)

    avg_loss, macro, per_ap, f1m, per_f1, per_cm = eval_one_ckpt(model, loader, device, num_classes, args.f1_thresh, keep_idx=keep_idx)

    print(f"\n[{Path(ckpt_path).name}] Loss: {avg_loss:.4f} | AUPRC(macro): {macro:.4f}")
    print(f"F1(macro @ {args.f1_thresh:.2f}): {f1m:.4f}")


    # Print a clean "table-like" set of labels (single values, no ±)
    table_labels = [
        "Lung Opacity",
        "Cardiomegaly",
        "Pleural thickening",
        "Aortic enlargement",
        "Pulmonary fibrosis",
        "Tuberculosis",
        "Pleural effusion",
    ]

    print("\nVinDr-CXR (AP/AUPRC) per class (single ckpt):")
    for name in table_labels:
        if name not in class_names:
            print(f"  {name}: MISSING from CSV header")
            continue
        i = class_names.index(name)
        m = 100.0 * float(per_ap[i])  # percent
        print(f"  {name:18s} {m:5.1f}")

    # Also print ALL classes (so you can paste into a full table if needed)
    print("\nAll classes (single ckpt) — in CSV header order:")
    for i, (name, apv, f1v) in enumerate(zip(class_names, per_ap, per_f1)):
        print(f"{name:25s} AP {100.0*float(apv):5.1f} | F1 {100.0*float(f1v):5.1f}")

    print(f"\nPer-class confusion matrices @ threshold={args.f1_thresh:.2f} (TN FP / FN TP):")
    for i, name in enumerate(class_names):
        tn, fp = per_cm[i, 0, 0], per_cm[i, 0, 1]
        fn, tp = per_cm[i, 1, 0], per_cm[i, 1, 1]
        print(f"{name:25s}  TN {tn:6d}  FP {fp:6d}  FN {fn:6d}  TP {tp:6d}")

    if args.wandb:
        wandb.log(
            {
                "loss": float(avg_loss),
                "auprc_macro": float(macro),
            }
        )
        # optional: log table-label APs
        for name in table_labels:
            if name in class_names:
                i = class_names.index(name)
                wandb.log({f"ap/{name}": float(per_ap[i])})
        wandb.finish()


if __name__ == "__main__":
    main()
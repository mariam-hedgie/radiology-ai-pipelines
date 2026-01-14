# scripts/eval_vindr_linear_probe.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb

from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelF1Score,   # <-- NEW
)

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
def eval_one_ckpt(model, loader, device, num_classes: int, f1_thresh: float):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    # --- AUPRC ---
    auprc_macro = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(device)
    auprc_per = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)

    # --- F1 (multilabel needs threshold) ---
    f1_micro = MultilabelF1Score(num_labels=num_classes, average="micro", threshold=f1_thresh).to(device)
    f1_macro = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=f1_thresh).to(device)
    f1_per = MultilabelF1Score(num_labels=num_classes, average=None, threshold=f1_thresh).to(device)

    loss_sum = 0.0
    n = 0

    auprc_macro.reset()
    auprc_per.reset()
    f1_micro.reset()
    f1_macro.reset()
    f1_per.reset()

    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        logits = model(imgs)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)

        # torchmetrics expects int targets for multilabel
        t_int = targets.int()

        auprc_macro.update(probs, t_int)
        auprc_per.update(probs, t_int)

        # F1 uses threshold internally
        f1_micro.update(probs, t_int)
        f1_macro.update(probs, t_int)
        f1_per.update(probs, t_int)

    avg_loss = loss_sum / max(1, n)

    auprc_macro_v = float(auprc_macro.compute().item())
    auprc_per_v = auprc_per.compute().detach().cpu().numpy().astype(np.float64)  # [K]

    f1_micro_v = float(f1_micro.compute().item())
    f1_macro_v = float(f1_macro.compute().item())
    f1_per_v = f1_per.compute().detach().cpu().numpy().astype(np.float64)        # [K]

    return avg_loss, auprc_macro_v, auprc_per_v, f1_micro_v, f1_macro_v, f1_per_v


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    ap.add_argument("--jepa_ckpt", type=str, default=None, help="Required when backbone=jepa")

    ap.add_argument("--dicom_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)

    # ONE checkpoint only (no ±)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)

    # NEW: F1 threshold
    ap.add_argument("--f1_thresh", type=float, default=0.5, help="Probability threshold for multilabel F1")

    # NEW: where to save CSVs
    ap.add_argument("--out_root", type=str, default="eval_outputs", help="Directory to save CSV summaries")

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="vindr-linear-probe")
    ap.add_argument("--wandb_name", type=str, default="eval")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Match preprocessing to backbone (IMPORTANT for iJEPA 448)
    if args.backbone == "ijepa":
        image_size = 448
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        image_size = args.image_size
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # dataset
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

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"

    # outputs
    out_dir = Path(args.out_root) / args.backbone / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

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
                "ckpt": str(ckpt_path),
                "f1_thresh": args.f1_thresh,
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

    # load head + eval
    load_head_only(model, str(ckpt_path))

    avg_loss, auprc_macro, auprc_per, f1_micro, f1_macro, f1_per = eval_one_ckpt(
        model, loader, device, num_classes, f1_thresh=args.f1_thresh
    )

    print(f"\n[{ckpt_path.name}] Loss: {avg_loss:.4f}")
    print(f"AUPRC(macro): {auprc_macro:.4f}")
    print(f"F1 micro @ {args.f1_thresh:.2f}: {f1_micro:.4f}")
    print(f"F1 macro @ {args.f1_thresh:.2f}: {f1_macro:.4f}")

    # table labels (paper-style subset)
    table_labels = [
        "Lung Opacity",
        "Cardiomegaly",
        "Pleural thickening",
        "Aortic enlargement",
        "Pulmonary fibrosis",
        "Tuberculosis",
        "Pleural effusion",
    ]

    print("\nVinDr-CXR (single ckpt) — subset classes:")
    print("  label                 AUPRC(%)   F1(%)")
    for name in table_labels:
        if name not in class_names:
            print(f"  {name:20s} MISSING")
            continue
        i = class_names.index(name)
        ap_pct = 100.0 * float(auprc_per[i])
        f1_pct = 100.0 * float(f1_per[i])
        print(f"  {name:20s} {ap_pct:8.1f} {f1_pct:7.1f}")

    # ---- SAVE CSVs ----
    per_csv = out_dir / "per_class_metrics.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "auprc", "f1"])
        for name, apv, f1v in zip(class_names, auprc_per, f1_per):
            w.writerow([name, float(apv), float(f1v)])

    summary_csv = out_dir / "summary_metrics.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["loss", float(avg_loss)])
        w.writerow(["auprc_macro", float(auprc_macro)])
        w.writerow(["f1_micro", float(f1_micro)])
        w.writerow(["f1_macro", float(f1_macro)])
        w.writerow(["f1_threshold", float(args.f1_thresh)])

    print(f"\nSaved per-class CSV: {per_csv}")
    print(f"Saved summary CSV:   {summary_csv}")

    if args.wandb:
        wandb.log(
            {
                "loss": float(avg_loss),
                "auprc_macro": float(auprc_macro),
                "f1_micro": float(f1_micro),
                "f1_macro": float(f1_macro),
                "f1_thresh": float(args.f1_thresh),
            }
        )
        # optional: log a few per-class
        for name in table_labels:
            if name in class_names:
                i = class_names.index(name)
                wandb.log(
                    {
                        f"auprc/{name}": float(auprc_per[i]),
                        f"f1/{name}": float(f1_per[i]),
                    }
                )
        wandb.finish()


if __name__ == "__main__":
    main()
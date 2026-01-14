import os
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
import wandb

from torchmetrics.classification import MultilabelAveragePrecision

from src.datasets.vindr_dataset import VinDrCXRImageLabels
from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone
from src.backbones.rad_jepa_backbone import build_rad_jepa_backbone
from src.backbones.ijepa_backbone import build_ijepa_backbone


# ------------------ utils ------------------
def save_state_dict(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()

    # backbone
    ap.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    ap.add_argument("--jepa_ckpt", type=str, default=None, help="Required if backbone=jepa")

    # data
    ap.add_argument("--train_dicom_dir", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)

    # training
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    # split
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # checkpoints
    ap.add_argument("--checkpoint_dir", type=str, default="None")
    ap.add_argument("--save_every", type=int, default=5)

    # device
    ap.add_argument("--device", type=str, default=None)

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="vindr-linear-probe")
    ap.add_argument("--wandb_name", type=str, default="")

    args = ap.parse_args()

    # device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, "best"), exist_ok=True)

    # ------------------ dataset ------------------
    if args.backbone == "ijepa":
        image_size = 448
        mean = (0.5, 0.5, 0.5)
        std  = (0.5, 0.5, 0.5)
    else:
        image_size = args.image_size
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    full_ds = VinDrCXRImageLabels(
        dicom_dir=args.train_dicom_dir,
        labels_csv=args.train_csv,
        image_size=image_size,
        mean=mean,
        std=std,
    ) 
    num_classes = full_ds.num_classes

    keep_labels = ["Lung Opacity", 
        "Cardiomegaly", 
        "Pleural thickening", 
        "Aortic enlargement",
        "Pulmonary fibrosis", 
        "Tuberculosis", 
        "Pleural effusion",
    ]

    missing = [ l for l in keep_labels if l not in full_ds.label_cols]
    if missing:
        raise ValueError(f"These labels are missing from CSV header: {missing}")

    keep_idx = [full_ds.label_cols.index(l) for l in keep_labels]

    # override num_classes to mtch the subset
    num_classes = len(keep_idx)
    print("Training on labels:", keep_labels)
    print("Labels indices:", keep_idx)

    # split
    g = torch.Generator().manual_seed(args.seed)
    n_total = len(full_ds)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------ wandb ------------------
    if args.wandb:
        run_name = args.wandb_name if args.wandb_name else (
            f"vindr-{args.backbone}-linear"
            f"-bs{args.batch_size}-lr{args.lr}"
            f"-vf{args.val_frac}-seed{args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "val_frac": args.val_frac,
                "seed": args.seed,
                "device": device,
                "encoder": f"RAD-{args.backbone.upper()} ViT-B (frozen)",
                "head": "Linear probe",
                "num_classes": num_classes,
            },
        )

    # ------------------ backbone ------------------
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)

    elif args.backbone == "jepa":
        if args.jepa_ckpt is None:
            raise ValueError("For --backbone jepa you must pass --jepa_ckpt /path/to/weights.pth.tar")
        backbone = build_rad_jepa_backbone(jepa_ckpt=args.jepa_ckpt, device=device)

    else:  # ijepa
        backbone = build_ijepa_backbone(device=device)

    # num_classes is 7
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)

    # freeze backbone
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    # loss + opt
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # metric
    auprc = MultilabelAveragePrecision(
        num_labels=num_classes,
        average="macro",
    ).to(device)

    best_val_auprc = -1.0
    best_val_loss = float("inf")

    # ------------------ training loop ------------------
    for epoch in range(args.epochs):
        # ---- TRAIN ----
        model.train()
        model.backbone.eval()

        train_loss_sum = 0.0
        n_train_seen = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)[:, keep_idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)
            n_train_seen += imgs.size(0)

        train_loss = train_loss_sum / max(1, n_train_seen)

        # ---- VAL ----
        model.eval()
        auprc.reset()
        val_loss_sum = 0.0
        n_val_seen = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)[:, keep_idx]

                logits = model(imgs)
                loss = criterion(logits, targets)

                probs = torch.sigmoid(logits)
                auprc.update(probs, targets.int())

                val_loss_sum += loss.item() * imgs.size(0)
                n_val_seen += imgs.size(0)

        val_loss = val_loss_sum / max(1, n_val_seen)
        val_auprc = auprc.compute().item()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUPRC(macro): {val_auprc:.4f}"
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/auprc_macro": val_auprc,
            })

        # save best
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            save_state_dict(model, os.path.join(args.checkpoint_dir, "best", "best_by_val_auprc.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_state_dict(model, os.path.join(args.checkpoint_dir, "best", "best_by_val_loss.pt"))

        if (epoch + 1) % args.save_every == 0:
            save_state_dict(model, os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"))

        save_state_dict(model, os.path.join(args.checkpoint_dir, "last.pt"))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
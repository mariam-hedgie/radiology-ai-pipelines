import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import wandb

from src.datasets.lung_dataset import LungSegDataset
from src.model import UPerNetSegModel


def save_ckpt(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_root", type=str, default="data/lung_seg")
    ap.add_argument("--num_classes", type=int, default=2)

    # backbone
    ap.add_argument("--backbone", type=str, required=True, choices=["dino", "jepa", "ijepa"])
    ap.add_argument("--jepa_ckpt", type=str, default=None)
    ap.add_argument("--ijepa_model_id", type=str, default="facebook/ijepa_vith16_1k")

    # training
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)

    # i/o
    ap.add_argument("--checkpoint_root", type=str, default="checkpoints_lungseg")
    ap.add_argument("--save_every", type=int, default=10)

    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="lung-seg")
    ap.add_argument("--wandb_name", type=str, default="")

    args = ap.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # ----- IMPORTANT: match input size to backbone -----
    # iJEPA vith16_1k expects 448x448 unless you explicitly interpolate pos enc.
    if args.backbone == "ijepa":
        image_size = 448
    else:
        image_size = 224

    # ----- checkpoint dir: isolate per backbone -----
    ckpt_dir = Path(args.checkpoint_root) / args.backbone
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- dataset -----
    # NOTE: LungSegDataset must resize BOTH image and mask to image_size.
    train_ds = LungSegDataset(
        img_dir=f"{args.data_root}/train/images",
        mask_dir=f"{args.data_root}/train/masks",
        image_size=image_size,   # <-- requires dataset support
    )
    val_ds = LungSegDataset(
        img_dir=f"{args.data_root}/val/images",
        mask_dir=f"{args.data_root}/val/masks",
        image_size=image_size,   # <-- requires dataset support
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- wandb -----
    if args.wandb:
        run_name = args.wandb_name or f"lungseg-{args.backbone}-bs{args.batch_size}-lr{args.lr}-sz{image_size}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "backbone": args.backbone,
                "image_size": image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_classes": args.num_classes,
                "jepa_ckpt": args.jepa_ckpt,
                "ijepa_model_id": args.ijepa_model_id,
            },
        )

    # ----- model -----
    model = UPerNetSegModel(
        num_classes=args.num_classes,
        backbone=args.backbone,
        jepa_ckpt=args.jepa_ckpt,
        ijepa_model_id=args.ijepa_model_id,
    ).to(device)

    # decoder-only training (backbone frozen inside backbone classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.decoder.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # ----- train loop -----
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / max(1, len(val_loader))

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if args.wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(model, ckpt_dir / "best.pth")
            print(f"  ↳ Saved BEST model (val loss = {best_val_loss:.4f})")
            if args.wandb:
                wandb.log({"best_val_loss": best_val_loss})

        # periodic
        if (epoch + 1) % args.save_every == 0:
            save_ckpt(model, ckpt_dir / f"epoch_{epoch+1}.pth")
            print(f"  ↳ Saved checkpoint at epoch {epoch+1}")

        # last
        save_ckpt(model, ckpt_dir / "last.pth")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
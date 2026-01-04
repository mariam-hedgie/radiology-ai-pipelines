import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import wandb

from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone


def build_loaders(data_root: str, image_size: int, batch_size: int, num_workers: int):
    tfm_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(root=os.path.join(data_root, "train"), transform=tfm_train)
    val_ds   = ImageFolder(root=os.path.join(data_root, "val"), transform=tfm_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes


def save_state_dict(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rad-dino-linear-probe")
    parser.add_argument("--wandb_name", type=str, default="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, "best"), exist_ok=True)

    # data
    train_loader, val_loader, classes = build_loaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(classes)

    # wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name if args.wandb_name else f"rad-dino-linear-bs{args.batch_size}-lr{args.lr}",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "encoder": "RAD-DINO ViT-B (frozen)",
                "head": "Linear probe",
                "dataset_root": args.data_root,
                "classes": classes,
            }
        )

    # model
    backbone = build_rad_dino_backbone(device=device)
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)

    # loss + optimizer (HEAD ONLY)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d} [Train]",
            leave=False
        ):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1:03d} [Val]",
                leave=False
            ):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        # ===== LOGGING =====
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "best/val_loss": best_val_loss,
                "best/val_acc": best_val_acc,
            })

        # ===== SAVE BEST (by val loss) =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_state_dict(model, os.path.join(args.checkpoint_dir, "best", "best_by_loss.pt"))
            if args.wandb:
                wandb.log({"best/val_loss": best_val_loss})
            print(f"  ↳ Saved BEST_BY_LOSS (val loss = {best_val_loss:.4f})")

        # ===== SAVE BEST (by val acc) =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_state_dict(model, os.path.join(args.checkpoint_dir, "best", "best_by_acc.pt"))
            if args.wandb:
                wandb.log({"best/val_acc": best_val_acc})
            print(f"  ↳ Saved BEST_BY_ACC (val acc = {best_val_acc:.4f})")

        # ===== SAVE PERIODIC =====
        if (epoch + 1) % args.save_every == 0:
            save_state_dict(model, os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"))
            print(f"  ↳ Saved checkpoint at epoch {epoch+1}")

        # ===== SAVE LAST =====
        save_state_dict(model, os.path.join(args.checkpoint_dir, "last.pt"))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
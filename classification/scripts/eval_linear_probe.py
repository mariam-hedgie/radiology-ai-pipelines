import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb

from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone


def build_loader(data_root: str, split: str, image_size: int, batch_size: int, num_workers: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    ds = ImageFolder(root=os.path.join(data_root, split), transform=tfm)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return ds, loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rad-dino-linear-probe")
    parser.add_argument("--wandb_name", type=str, default="eval")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    ds, loader = build_loader(
        data_root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(ds.classes)

    # wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "split": args.split,
                "ckpt": args.ckpt,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "dataset_root": args.data_root,
                "classes": ds.classes,
                "encoder": "RAD-DINO ViT-B (frozen)",
                "head": "Linear probe",
            }
        )

    # model
    backbone = build_rad_dino_backbone(device=device)
    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    # Support BOTH formats:
    # 1) checkpoint dict with "model" key (if you saved full ckpt)
    # 2) raw state_dict (if you saved model.state_dict())
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Eval [{args.split}]", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            loss_sum += loss.item() * imgs.size(0)

    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)

    print(f"[{args.split}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    if args.wandb:
        wandb.log({
            f"{args.split}/loss": avg_loss,
            f"{args.split}/acc": acc
        })
        wandb.finish()


if __name__ == "__main__":
    main()
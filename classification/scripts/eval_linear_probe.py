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
from src.backbones.rad_jepa_backbone import build_rad_jepa_backbone


def build_loader(data_root: str, split: str, image_size: int, batch_size: int, num_workers: int):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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

    parser.add_argument("--backbone", type=str, default="dino", choices=["dino", "jepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Path to JEPA pretrained weights")

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rad-linear-probe")
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
                "encoder": f"RAD-{args.backbone.upper()} ViT-B (frozen)",
                "head": "Linear probe",
            }
        )

    # model
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)
    else:
        if args.jepa_ckpt is None:
            raise ValueError("For --backbone jepa you must pass --jepa_ckpt /path/to/weights.pth.tar")
        backbone = build_rad_jepa_backbone(jepa_ckpt=args.jepa_ckpt, device=device)

    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # unwrap common checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 1) normalize keys: remove "module." / "model." prefixes
    norm = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        norm[k] = v

    # 2) extract ONLY classifier weights
    head_state = {}
    for k, v in norm.items():
        if k.startswith("classifier."):
            head_state[k[len("classifier."):]] = v

    if len(head_state) == 0:
        raise RuntimeError(
            f"No classifier.* keys found in checkpoint {args.ckpt}. "
            f"Available keys (first 30): {list(norm.keys())[:30]}"
        )

    # 3) load head strictly (this should NOT be flexible)
    model.classifier.load_state_dict(head_state, strict=True)
    print(f"Loaded linear head from: {args.ckpt} (keys: {list(head_state.keys())})")

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
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.models.linear_probe import LinearProbeClassifier
from src.backbones.rad_dino_backbone import build_rad_dino_backbone
from src.backbones.rad_jepa_backbone import build_rad_jepa_backbone


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def pil_to_tensor_resize(img: Image.Image, size: int) -> torch.Tensor:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0      # [H,W,3] in [0,1]
    arr = np.transpose(arr, (2, 0, 1))                 # -> [3,H,W]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD         # normalize
    return torch.from_numpy(arr)


class FolderClassDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], image_size: int):
        self.root = root
        self.classes = classes
        self.image_size = image_size
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        samples = []
        for c in classes:
            class_dir = root / c
            if not class_dir.exists():
                continue
            for ext in IMG_EXTS:
                for p in class_dir.glob(f"*{ext}"):
                    samples.append((p, self.class_to_idx[c]))

        samples.sort(key=lambda x: str(x[0]))
        if len(samples) == 0:
            raise RuntimeError(f"No images found under: {root}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path)
        x = pil_to_tensor_resize(img, self.image_size)
        return x, label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_precision_recall_f1(cm: np.ndarray):
    # cm[t, p]
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp

        precision[k] = tp / (tp + fp + 1e-12)
        recall[k] = tp / (tp + fn + 1e-12)
        f1[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k] + 1e-12)

    return precision, recall, f1


def binary_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    y_true: {0,1}
    y_score: probability for class 1
    Computes ROC-AUC without sklearn.
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return float("nan")

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)

    # trapezoid area
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dino", choices=["dino", "jepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Path to JEPA pretrained weights (.pth/.pth.tar)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt", type=str, default="checkpoints/best/best_by_acc.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = Path(args.data_root) / args.split
    assert split_dir.exists(), f"Missing: {split_dir}"
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    num_classes = len(classes)
    assert num_classes >= 2, f"Need >=2 classes, found {classes}"

    ds = FolderClassDataset(split_dir, classes=classes, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    # ---- build backbone ----
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)
    else:
        if args.jepa_ckpt is None:
            raise ValueError("For --backbone jepa you must pass --jepa_ckpt /path/to/best_jepa_weights.pth.tar")
        backbone = build_rad_jepa_backbone(jepa_ckpt=args.jepa_ckpt, device=device)

    model = LinearProbeClassifier(backbone=backbone, num_classes=num_classes).to(device)

    # freeze backbone (safety)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ---- load ONLY the linear head from checkpoint ----
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

    # extract classifier.* weights only
    head_state = {}
    for k, v in norm.items():
        if k.startswith("classifier."):
            head_state[k[len("classifier."):]] = v

    if len(head_state) == 0:
        raise RuntimeError(
            f"No classifier.* keys found in checkpoint {ckpt_path}. "
            f"Keys (first 30): {list(norm.keys())[:30]}"
        )

    model.classifier.load_state_dict(head_state, strict=True)
    model.eval()

    all_true = []
    all_pred = []
    all_probs = []
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    acc = correct / max(1, total)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    precision, recall, f1 = per_class_precision_recall_f1(cm)

    macro_f1 = float(np.mean(f1))
    balanced_acc = float(np.mean(recall))  # mean recall across classes

    # ROC-AUC (only do it cleanly for binary)
    auc = None
    if num_classes == 2:
        auc = binary_auc_from_scores(y_true, y_prob[:, 1])

    print("\n=== Linear Probe Evaluation ===")
    print(f"Split: {args.split} | N={len(ds)} | Classes={classes}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Balanced Acc:    {balanced_acc:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC:         {auc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nPer-class metrics:")
    for i, c in enumerate(classes):
        print(f"  {c:12s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1[i]:.4f}")

    print()


if __name__ == "__main__":
    main()
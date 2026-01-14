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
from src.backbones.ijepa_backbone import build_ijepa_backbone


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ------------------ preprocessing ------------------
def pil_to_tensor_resize(
    img: Image.Image,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean = np.array(mean, dtype=np.float32)[:, None, None]
    std  = np.array(std,  dtype=np.float32)[:, None, None]

    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0      # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))                 # [3,H,W]
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


# ------------------ dataset ------------------
class FolderClassDataset(Dataset):
    def __init__(
        self,
        root: Path,
        classes: list[str],
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        self.root = root
        self.classes = classes
        self.image_size = image_size
        self.mean = mean
        self.std = std
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
        x = pil_to_tensor_resize(
            img,
            size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        return x, label


def collate_fn(batch):
    imgs, labels = zip(*batch) # pair by position
    return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long)


# ------------------ metrics ------------------
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_precision_recall_f1(cm):
    k = cm.shape[0]
    p = np.zeros(k)
    r = np.zeros(k)
    f = np.zeros(k)

    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p[i] = tp / (tp + fp + 1e-12)
        r[i] = tp / (tp + fn + 1e-12)
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i] + 1e-12)

    return p, r, f


def binary_auc_from_scores(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return float("nan")

    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]

    for yt in y_true:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)

    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc

def binary_auprc_from_scores(y_true, y_score):
    """
    PR-AUC for binary classifciation. 
    y_true: {0,1}
    y_score: probability for class1
    """
    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = y_true.sum()
    if P == 0:
        return float("nan")

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for yt in y_true:
        if yt == 1:
            tp+=1
        else:
            fp+=1
        prec = tp/(tp+fp+1e-12)
        rec = tp/(P+1e-12)

        precisions.append(prec)
        recalls.append(rec)
    
    # area udnder PR curve - step integral
    auprc = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        auprc += p*(r-prev_r)
        prev_r = r

    return auprc


# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dino", choices=["dino", "jepa", "ijepa"])
    parser.add_argument("--jepa_ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ preprocessing policy ------------------
    if args.backbone == "ijepa":
        image_size = 448 if args.image_size == 224 else args.image_size
        mean = (0.5, 0.5, 0.5)
        std  = (0.5, 0.5, 0.5)
    else:
        image_size = args.image_size
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    split_dir = Path(args.data_root) / args.split
    assert split_dir.exists(), split_dir
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), ckpt_path

    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    num_classes = len(classes)
    print("class order (label mapping):", {c: i for i, c in enumerate(classes)})

    ds = FolderClassDataset(
        split_dir,
        classes=classes,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ------------------ backbone ------------------
    if args.backbone == "dino":
        backbone = build_rad_dino_backbone(device=device)
    elif args.backbone == "jepa":
        if args.jepa_ckpt is None:
            raise ValueError("Missing --jepa_ckpt")
        backbone = build_rad_jepa_backbone(args.jepa_ckpt, device=device)
    else:
        backbone = build_ijepa_backbone(device=device)

    model = LinearProbeClassifier(backbone, num_classes).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ------------------ load head only ------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    head_state = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("model.", "")
        if k.startswith("classifier."):
            head_state[k[len("classifier."):]] = v

    model.classifier.load_state_dict(head_state, strict=True)
    model.eval()

    # ------------------ eval ------------------
    all_true, all_pred, all_probs = [], [], []
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    acc = correct / total
    cm = confusion_matrix(y_true, y_pred, num_classes)
    p, r, f1 = per_class_precision_recall_f1(cm)

    print("\n=== Linear Probe Evaluation ===")
    print(f"Backbone: {args.backbone}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1.mean():.4f}")

    if num_classes == 2:
        pos_scores = y_prob[:, 1]

        roc_auc = binary_auc_from_scores(y_true, pos_scores)
        auprc = binary_auprc_from_scores(y_true, pos_scores)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
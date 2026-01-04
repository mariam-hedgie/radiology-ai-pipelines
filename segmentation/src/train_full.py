import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import wandb

from src.datasets.lung_dataset import LungSegDataset
#from src.model import RadDinoUPerNet
from src.model import UPerNetSegModel

# ---------------- CONFIG ----------------
DATA_ROOT = "data/lung_seg"
NUM_CLASSES = 2
BATCH_SIZE = 2
NUM_EPOCHS = 300
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints_jepa"
SAVE_EVERY = 10
# ---------------------------------------


def main():
    # ---------- W&B INIT ----------
    wandb.init(
        project="rad-dino-lung-seg",
        name=f"rad-dino-upernet-bs{BATCH_SIZE}-lr{LR}",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "encoder": "RAD-DINO ViT-B",
            "decoder": "UPerNet",
            "dataset": "Kaggle Lung Segmentation"
        }
    )

    # ---------- Setup ----------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- Datasets ----------
    train_ds = LungSegDataset(
        img_dir=f"{DATA_ROOT}/train/images",
        mask_dir=f"{DATA_ROOT}/train/masks"
    )

    val_ds = LungSegDataset(
        img_dir=f"{DATA_ROOT}/val/images",
        mask_dir=f"{DATA_ROOT}/val/masks"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False,   # ← MPS doesn’t use this
        drop_last=True      # ← CRITICAL FIX
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False    # ← optional but cleaner on Mac
    )

    # ---------- Model ----------
    #model = RadDinoUPerNet(num_classes=NUM_CLASSES)
    model = UPerNetSegModel(
    num_classes=NUM_CLASSES,
    backbone="jepa",
    jepa_ckpt="/data1/mariam/best_jepa_weights.pth.tar"
)

    model.to(DEVICE)

    # ---------- Loss & Optimizer ----------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.decoder.parameters(),  # decoder only
        lr=LR
    )

    best_val_loss = float("inf")

    # ---------- Training Loop ----------
    for epoch in range(NUM_EPOCHS):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d} [Train]",
            leave=False
        ):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1:03d} [Val]",
                leave=False
            ):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # ===== LOGGING =====
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # ===== SAVE BEST =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best.pth")
            )
            wandb.log({"best_val_loss": best_val_loss})
            print(f"  ↳ Saved BEST model (val loss = {best_val_loss:.4f})")

        # ===== SAVE PERIODIC =====
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    CHECKPOINT_DIR,
                    f"epoch_{epoch+1}.pth"
                )
            )
            print(f"  ↳ Saved checkpoint at epoch {epoch+1}")

    wandb.finish()


if __name__ == "__main__":
    main()
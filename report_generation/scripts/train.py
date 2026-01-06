# scripts/train.py
import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    get_cosine_schedule_with_warmup
)

from src.config import TrainConfig
from src.utils.seed import set_seed
from src.utils.ckpt import save_ckpt
from src.data_functions.dataset import JsonlImageTextDataset
from src.data_functions.collate import CollateImageText
from src.models.report_generator import VisionLLMReportGenerator
from src.models.rad_dino_encoder import FrozenRadDinoEncoder
from src.models.rad_jepa_encoder import FrozenRadJepaEncoder


def main():
    parser = argparse.ArgumentParser()

    # data/model ids
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--backbone", type=str, required=True, choices=["rad-dino", "rad-jepa"])
    parser.add_argument("--vision_id", type=str, required=True,
                        help="HF model id for vision encoder (RAD-DINO or RAD-JEPA)")
    parser.add_argument("--llm_name", type=str, default=None)

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="rad-dino-report-gen")
    parser.add_argument("--run_name", type=str, default="debug")

    args = parser.parse_args()

    # ---------------- config ----------------
    cfg = TrainConfig()
    if args.data_root: cfg.data_root = args.data_root
    if args.llm_name: cfg.llm_name = args.llm_name
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.grad_accum: cfg.grad_accum = args.grad_accum
    if args.warmup_ratio is not None: cfg.warmup_ratio = args.warmup_ratio
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.save_every_steps is not None: cfg.save_every_steps = args.save_every_steps

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ---------------- wandb ----------------
    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                **cfg.__dict__,
                "backbone": args.backbone,
                "vision_id": args.vision_id,
            }
        )

    # ---------------- tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- data ----------------
    train_ds = JsonlImageTextDataset(cfg.data_root, cfg.train_jsonl)
    val_ds   = JsonlImageTextDataset(cfg.data_root, cfg.val_jsonl)

    collate = CollateImageText(
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        image_size=cfg.image_size,
        max_text_len=cfg.max_text_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    # make modular
    image_processor = AutoImageProcessor.from_pretrained(args.vision_id)
    vision_base = AutoModel.from_pretrained(args.vision_id, trust_remote_code=True)

    if args.backbone == "rad-dino":
        vision = FrozenRadDinoEncoder(
            rad_dino_model=vision_base,
            image_processor=image_processor
        ).to(device)

    elif args.backbone == "rad-jepa":
        vision = FrozenRadJepaEncoder(
            rad_jepa_model=vision_base,
            image_processor=image_processor
        ).to(device)

    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # ---- sanity check: vision must return [B,N,D] ----
    vision.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device, dtype=torch.float32)
        pt = vision(dummy)
        assert pt.ndim == 3, f"Expected [B,N,D] tokens, got {tuple(pt.shape)}"
        assert pt.shape[-1] == 768, f"Projector expects 768-dim tokens, got D={pt.shape[-1]}"
        print("vision tokens:", tuple(pt.shape))

    # ---------------- full model ----------------
    model = VisionLLMReportGenerator(
        vision_encoder=vision,
        llm_name=cfg.llm_name
    ).to(device)

    # Freeze vision (paper)
    model.freeze_vision()

    # ---------------- optimizer (projector + LLM) ----------------
    # Paper trains projector + LLM, with vision frozen.
    params = list(model.projector.parameters()) + list(model.llm.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------- schedule (cosine + 3% warmup) ----------------
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ---------------- training ----------------
    best_val = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]")

        for step, batch in enumerate(pbar):
            # move tensors to device (collate returns tensors)
            batch["images"] = batch["images"].to(device, non_blocking=True)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["attention_mask"] = batch["attention_mask"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)

            loss = model(batch) / cfg.grad_accum
            loss.backward()

            running_epoch_loss += loss.item() * cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix({"loss": f"{running_epoch_loss/(step+1):.4f}"})

                if args.wandb:
                    wandb.log({
                        "step": global_step,
                        "train/loss": running_epoch_loss / (step + 1),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1
                    })

                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    save_ckpt(
                        os.path.join(cfg.ckpt_dir, f"step_{global_step}.pt"),
                        model, optimizer, global_step, best_val
                    )

        # ---------------- validation ----------------
        model.eval()
        val_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [val]"):
                batch["images"] = batch["images"].to(device, non_blocking=True)
                batch["prompt_input_ids"] = batch["prompt_input_ids"].to(device, non_blocking=True)
                batch["target_input_ids"] = batch["target_input_ids"].to(device, non_blocking=True)

                loss = model(batch, tokenizer)
                val_loss += loss.item()
                n += 1

        val_loss = val_loss / max(1, n)
        print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")

        if args.wandb:
            wandb.log({"val/loss": val_loss, "epoch": epoch + 1})

        # ---------------- save best ----------------
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(
                os.path.join(cfg.ckpt_dir, "best.pt"),
                model, optimizer, global_step, best_val
            )
            print(f"Saved BEST checkpoint (val_loss={best_val:.4f})")

        # save last each epoch
        save_ckpt(
            os.path.join(cfg.ckpt_dir, "last.pt"),
            model, optimizer, global_step, best_val
        )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
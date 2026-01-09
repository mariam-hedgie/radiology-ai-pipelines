# scripts/train.py
import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from transformers import AutoModel, AutoImageProcessor, get_cosine_schedule_with_warmup

from src.config import TrainConfig
from src.utils.seed import set_seed
from src.utils.ckpt import save_ckpt
from src.data_functions.dataset import JsonlImageTextDataset
from src.data_functions.collate import CollateImageText
from src.models.report_generator import VisionLLMReportGenerator
from src.models.rad_dino_encoder import FrozenRadDinoEncoder
from src.models.rad_jepa_encoder import FrozenRadJepaEncoder
from src.models.hf_ijepa_encoder import FrozenHFIJEPAEncoder



def _get_llm_device(model: VisionLLMReportGenerator) -> torch.device:
    # robust for device_map="auto"
    emb = model.llm.get_input_embeddings()
    return emb.weight.device


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--backbone", type=str, required=True, choices=["rad-dino", "rad-jepa", "ijepa-hf"])
    parser.add_argument("--vision_id", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Path to local JEPA .pth.tar")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="rad-report-gen")
    parser.add_argument("--run_name", type=str, default="debug")

    args = parser.parse_args()

    # ---------------- config ----------------
    cfg = TrainConfig()
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.llm_name is not None:
        cfg.llm_name = args.llm_name
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.warmup_ratio is not None:
        cfg.warmup_ratio = args.warmup_ratio
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.save_every_steps is not None:
        cfg.save_every_steps = args.save_every_steps

    # grad accum compatibility: cfg.grad_accum_steps vs cfg.grad_accum
    cfg_grad_accum = getattr(cfg, "grad_accum", None)
    if cfg_grad_accum is None:
        cfg_grad_accum = getattr(cfg, "grad_accum_steps", 1)
    if args.grad_accum is not None:
        grad_accum = args.grad_accum
    else:
        grad_accum = int(cfg_grad_accum)

    ckpt_root = str(cfg.ckpt_dir)
    os.makedirs(ckpt_root, exist_ok=True)

    set_seed(getattr(cfg, "seed", 42))

    vision_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- wandb ----------------
    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                **cfg.__dict__,
                "backbone": args.backbone,
                "vision_id": args.vision_id,
                "grad_accum": grad_accum,
            },
        )

    # ---------------- data ----------------
    train_ds = JsonlImageTextDataset(cfg.data_root, cfg.train_jsonl)
    val_ds = JsonlImageTextDataset(cfg.data_root, cfg.val_jsonl)

    # NOTE: tokenizer comes from the model (QLoRA builder) — but collate needs it.
    # We'll build the vision + model first, then build collate/loaders.

    # change image size to 448x448 if IJEPA
    if args.backbone == "ijepa-hf":
        cfg.image_size = 448

    # ---------------- vision backbone ----------------
    if args.backbone == "rad-dino":
        if args.vision_id is None:
            raise ValueError("--vision_id is required for rad-dino")

        image_processor = AutoImageProcessor.from_pretrained(
            args.vision_id,
            trust_remote_code=True
        )
        vision_base = AutoModel.from_pretrained(
            args.vision_id,
            trust_remote_code=True
        )
        vision = FrozenRadDinoEncoder(
            vision_base,
            image_processor=image_processor
        ).to(vision_device)

    elif args.backbone == "rad-jepa":
        if args.jepa_ckpt is None:
            raise ValueError("--jepa_ckpt is required for rad-jepa (local checkpoint)")

        # NOTE: FrozenRadJepaEncoder must be the "local checkpoint" version:
        # FrozenRadJepaEncoder(ckpt_path=..., image_size=...)
        vision = FrozenRadJepaEncoder(
            ckpt_path=args.jepa_ckpt,
            image_size=cfg.image_size,
        ).to(vision_device)
    
    
    elif args.backbone == "ijepa-hf":
        if args.vision_id is None:
            args.vision_id = "facebook/ijepa_vith16_1k"
        image_processor = AutoImageProcessor.from_pretrained(args.vision_id)
        vision_base = AutoModel.from_pretrained(args.vision_id)
        vision = FrozenHFIJEPAEncoder(vision_base, image_processor).to(vision_device)

    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # sanity check
    vision.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=vision_device, dtype=torch.float32)
        pt = vision(dummy)
        assert pt.ndim == 3, f"Expected [B,N,D] tokens, got {tuple(pt.shape)}"
        assert pt.shape[-1] == vision.embed_dim, f"Expected D={vision.embed_dim}, got D={pt.shape[-1]}"
        print("vision tokens:", tuple(pt.shape))

    # ---------------- full model (DO NOT .to(device) for QLoRA) ----------------
    # Your report_generator should internally build QLoRA if configured.
    model = VisionLLMReportGenerator(
        vision_encoder=vision,
        llm_name=cfg.llm_name,
        use_qlora=True,
        qlora_kwargs=dict(
            load_in_4bit=True,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            use_gradient_checkpointing=True,
        ),
    )

    # freeze vision
    model.freeze_vision()

    # tokenizer from model if you stored it; otherwise you must still load it outside
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        # fallback if your generator doesn't store tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    collate = CollateImageText(
        tokenizer=tokenizer,
        prompt=cfg.prompt,
        image_size=cfg.image_size,
        max_text_len=cfg.max_text_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # ---------------- optimizer: ONLY trainable params ----------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------- schedule ----------------
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    llm_device = _get_llm_device(model)
    print("LLM embedding device:", llm_device)
    print("Vision device:", vision_device)

    # ---------------- training ----------------
    best_val = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]")

        for step, batch in enumerate(pbar):
            # IMPORTANT: only move images; ids can stay on CPU
            batch["images"] = batch["images"].to(vision_device, non_blocking=True)

            loss = model(batch) / grad_accum
            loss.backward()

            running += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                try:
                    torch.nn.utils.clip_grad_norm_(trainable_params, getattr(cfg, "max_grad_norm", 1.0))
                except RuntimeError:
                    pass
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

                if args.wandb:
                    wandb.log(
                        {
                            "step": global_step,
                            "train/loss": running / (step + 1),
                            "lr": scheduler.get_last_lr()[0],
                            "depoch": epoch + 1,
                        }
                    )

                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    save_ckpt(
                        os.path.join(ckpt_root, f"step_{global_step}"),
                        model,
                        optimizer,
                        global_step,
                        best_val,
                    )

        # flush partial accumulation
        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, getattr(cfg, "max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # ---------------- validation ----------------
        model.eval()
        val_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [val]"):
                batch["images"] = batch["images"].to(vision_device, non_blocking=True)
                loss = model(batch)
                val_loss += float(loss.item())
                n += 1

        val_loss /= max(1, n)
        print(f"Epoch {epoch+1}: val_loss = {val_loss:.4f}")

        if args.wandb:
            wandb.log({"val/loss": val_loss, "epoch": epoch + 1})

        # ---------------- save best/last (DIRECTORIES) ----------------
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(os.path.join(ckpt_root, "best"), model, optimizer, global_step, best_val)
            print(f"Saved BEST checkpoint (val_loss={best_val:.4f})")

        save_ckpt(os.path.join(ckpt_root, "last"), model, optimizer, global_step, best_val)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
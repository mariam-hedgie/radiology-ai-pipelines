# scripts/generate.py
import os
import json
import argparse
import torch
from tqdm import tqdm

import wandb
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

from src.config import TrainConfig
from src.utils.ckpt import load_ckpt
from src.data.dataset import JsonlImageTextDataset
from src.data.collate import CollateImageText
from src.models.report_generator import RadDinoReportGenerator
from src.models.rad_dino_encoder import FrozenRadDinoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/dummy")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--rad_dino_id", type=str, required=True,
                        help="HuggingFace model id for RAD-DINO encoder")
    parser.add_argument("--out", type=str, default="outputs/preds.jsonl")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="rad-dino-report-gen")
    parser.add_argument("--run_name", type=str, default="generate")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_root = args.data_root
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- wandb (optional) ----
    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "data_root": args.data_root,
                "split": args.split,
                "ckpt": args.ckpt,
                "rad_dino_id": args.rad_dino_id,
                "llm_name": cfg.llm_name,
                "prompt": cfg.prompt,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "max_samples": args.max_samples,
            }
        )

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- dataset ----
    jsonl_name = {"train": cfg.train_jsonl, "val": cfg.val_jsonl, "test": getattr(cfg, "test_jsonl", cfg.val_jsonl)}[args.split]
    ds = JsonlImageTextDataset(cfg.data_root, jsonl_name=jsonl_name)

    collate = CollateImageText(tokenizer, cfg.prompt, cfg.image_size, cfg.max_text_len)

    # ---- build RAD-DINO vision encoder (frozen) ----
    image_processor = AutoImageProcessor.from_pretrained(args.rad_dino_id)
    rad_dino = AutoModel.from_pretrained(args.rad_dino_id)

    vision = FrozenRadDinoEncoder(rad_dino_model=rad_dino, image_processor=image_processor).to(device)

    # ---- build full model ----
    model = RadDinoReportGenerator(vision_encoder=vision, llm_name=cfg.llm_name).to(device)
    model.freeze_vision()

    # ---- load checkpoint ----
    load_ckpt(args.ckpt, model, optimizer=None, map_location="cpu")
    model.eval()

    # ---- prompt ids (batch size 1 for generation) ----
    prompt_ids = tokenizer([cfg.prompt], return_tensors="pt", padding=False).input_ids.to(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = min(args.max_samples, len(ds))
    print(f"Generating on {n} samples from split={args.split}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Saving to: {args.out}")

    rows = []
    with torch.no_grad():
        for i in tqdm(range(n)):
            batch = collate([ds[i]])  # single example
            imgs = batch["images"].to(device)

            gen_ids = model.generate(
                images=imgs,
                prompt_ids=prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature
            )

            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            rec = {
                "index": i,
                "path": batch["paths"][0],
                "prompt": cfg.prompt,
                "target": batch["raw_targets"][0],
                "gen": gen_text,
            }
            rows.append(rec)

            # optional: log a few examples to wandb
            if args.wandb and i < 10:
                wandb.log({f"sample/{i}": wandb.Table(data=[[rec["path"], rec["target"], rec["gen"]]],
                                                     columns=["path", "target", "gen"])})

    # write jsonl
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(rows)} generations.")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
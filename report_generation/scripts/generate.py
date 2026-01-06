# scripts/generate.py
import os
import json
import argparse
import torch
from tqdm import tqdm
import wandb

from transformers import AutoModel, AutoImageProcessor

from src.config import TrainConfig
from src.utils.ckpt import load_ckpt
from src.data_functions.dataset import JsonlImageTextDataset
from src.data_functions.collate import CollateImageText
from src.models.report_generator import VisionLLMReportGenerator
from src.models.rad_dino_encoder import FrozenRadDinoEncoder
from src.models.rad_jepa_encoder import FrozenRadJepaEncoder


def _get_llm_device(model: VisionLLMReportGenerator) -> torch.device:
    emb = model.llm.get_input_embeddings()
    return emb.weight.device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/iu_xray")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory (e.g. checkpoints/.../best)")
    parser.add_argument("--backbone", type=str, required=True, choices=["rad-dino", "rad-jepa"])
    parser.add_argument("--vision_id", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/preds.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--jepa_ckpt", type=str, default=None, help="Path to local JEPA .pth.tar (required for rad-jepa)")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="rad-report-gen")
    parser.add_argument("--run_name", type=str, default="generate")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_root = args.data_root
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature

    vision_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ----
    jsonl_name = {"train": cfg.train_jsonl, "val": cfg.val_jsonl, "test": cfg.test_jsonl}[args.split]
    ds = JsonlImageTextDataset(cfg.data_root, jsonl_name=jsonl_name)

    # ---- build vision encoder ----
    if args.backbone == "rad-dino":
        if args.vision_id is None:
            raise ValueError("--vision_id is required for rad-dino")

        image_processor = AutoImageProcessor.from_pretrained(args.vision_id, trust_remote_code=True)
        vision_base = AutoModel.from_pretrained(args.vision_id, trust_remote_code=True)
        vision = FrozenRadDinoEncoder(vision_base, image_processor=image_processor).to(vision_device)

    elif args.backbone == "rad-jepa":
        if args.jepa_ckpt is None:
            raise ValueError("--jepa_ckpt is required for rad-jepa (local checkpoint)")

        # IMPORTANT: your rad_jepa_encoder.py must support ckpt_path
        vision = FrozenRadJepaEncoder(
            ckpt_path=args.jepa_ckpt
        ).to(vision_device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # ---- build full model (DO NOT .to(device)) ----
    model = VisionLLMReportGenerator(
    vision_encoder=vision,
    llm_name=cfg.llm_name,
    use_qlora=True,
    qlora_kwargs={
        "load_in_4bit": cfg.load_in_4bit,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": cfg.lora_target_modules,
        "use_gradient_checkpointing": False,  # generation
    },
)
    model.freeze_vision()

    # ---- load adapters (your load_ckpt should attach to model.llm or whole wrapper) ----
    model, _ = load_ckpt(args.ckpt, model, optimizer=None, map_location="cpu")
    model.eval()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    collate = CollateImageText(tokenizer, cfg.prompt, cfg.image_size, cfg.max_text_len)

    llm_device = _get_llm_device(model)
    print("LLM embedding device:", llm_device)
    print("Vision device:", vision_device)

    # ---- prompt ids (on LLM device) ----
    prompt_ids = tokenizer(cfg.prompt, return_tensors="pt", padding=False).input_ids.to(llm_device)
    prompt_attn = torch.ones_like(prompt_ids, device=llm_device)
    prompt_len = prompt_ids.shape[1]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = min(args.max_samples, len(ds))
    print(f"Generating on {n} samples from split={args.split}")
    print(f"Checkpoint dir: {args.ckpt}")
    print(f"Saving to: {args.out}")

    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "data_root": args.data_root,
                "split": args.split,
                "ckpt": args.ckpt,
                "backbone": args.backbone,
                "vision_id": args.vision_id,
                "llm_name": cfg.llm_name,
                "prompt": cfg.prompt,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "max_samples": n,
            },
        )

    rows = []
    with torch.no_grad():
        for i in tqdm(range(n)):
            batch = collate([ds[i]])  # single item batch
            imgs = batch["images"].to(vision_device)

            gen_ids = model.generate(
                images=imgs,
                input_ids=prompt_ids,
                attention_mask=prompt_attn,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )

          
            full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            if full_text.startswith(cfg.prompt):
                gen_text = full_text[len(cfg.prompt):].strip()
            else:
                gen_text = full_text

            rec = {
                "index": i,
                "path": batch["paths"][0],
                "prompt": cfg.prompt,
                "target": batch["raw_targets"][0],
                "gen": gen_text,
            }
            rows.append(rec)

            if args.wandb and i < 10:
                wandb.log(
                    {
                        f"sample/{i}": wandb.Table(
                            data=[[rec["path"], rec["target"], rec["gen"]]],
                            columns=["path", "target", "gen"],
                        )
                    }
                )

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(rows)} generations -> {args.out}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
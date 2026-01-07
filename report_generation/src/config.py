from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # --------------------
    # Data layout
    # data/<split>/images
    # data/<split>/<split>.jsonl 
    # --------------------
    data_root: Path = Path("data")
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    images_dirname: str = "images"
    train_jsonl: str = "train.jsonl"
    val_jsonl: str = "val.jsonl"
    test_jsonl: str = "test.jsonl"

    # --------------------
    # Prompt + text lengths
    # --------------------
    prompt: str = "Provide a description of the findings in the radiology image."
    max_text_len: int = 512          # can try 256 if this doesn't work
    max_new_tokens: int = 150        # paper inference length
    temperature: float = 0.7

    # --------------------
    # Vision encoder (RAD-DINO ViT-B)
    # --------------------
    image_size: int = 224         
    vision_embed_dim: int = 768
    freeze_vision: bool = True

    # --------------------
    # LLM + QLoRA
    # --------------------
    llm_name: str = "lmsys/vicuna-7b-v1.5"

    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"   # switch to "float16" if bf16 unsupported

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # --------------------
    # Training for toy dataset
    # --------------------
    epochs: int = 3                  
    batch_size: int = 1              # start 1, increase later
    grad_accum: int = 4        # effective batch ~4
    lr: float = 2e-5                 # paper LR
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_workers: int = 2

    # --------------------
    # Checkpointing / logging
    # --------------------
    ckpt_dir: Path = Path("checkpoints")
    save_every_steps: int = 200
    eval_every_steps: int = 200

    use_wandb: bool = False
    wandb_project: str = "rad-dino-report-gen"
    wandb_run_name: str = "debug"

    seed: int = 42
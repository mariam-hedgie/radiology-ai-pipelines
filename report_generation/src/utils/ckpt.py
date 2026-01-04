# src/utils/ckpt.py
import os
import torch

def save_ckpt(path, model, optimizer, step, best_val=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
    }, path)

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
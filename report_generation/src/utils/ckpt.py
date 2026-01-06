# src/utils/ckpt.py
import os
import torch
from peft import PeftModel

def save_ckpt(path, model, optimizer=None, step=None, best_val=None):
    os.makedirs(path, exist_ok=True)

    # saves: adapter_model.safetensors + adapter_config.json
    model.save_pretrained(path)

    torch.save(
        {
            "step": step,
            "best_val": best_val,
            "optimizer": optimizer.state_dict() if optimizer else None,
        },
        os.path.join(path, "trainer_state.pt"),
    )

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    """
    path: directory produced by save_ckpt(), containing adapter files + trainer_state.pt
    model: the *base* model already loaded from_pretrained (quantized or not)
    returns: (model_with_adapters, state_dict_or_None)
    """

    # 1) attach adapters
    model = PeftModel.from_pretrained(
        model,
        path,
        is_trainable=(optimizer is not None),
    )

    # 2) load trainer state (optional)
    state_path = os.path.join(path, "trainer_state.pt")
    state = None
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=map_location)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])

    return model, state
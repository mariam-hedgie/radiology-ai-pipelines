# src/utils/ckpt.py
import os
import torch
from peft import PeftModel


def _unwrap_base_llm(llm):
    """
    PeftModel.from_pretrained expects a *base* model, not an already PEFT-wrapped model.
    """
    if isinstance(llm, PeftModel):
        # best-effort unwrap
        if hasattr(llm, "get_base_model"):
            return llm.get_base_model()
        if hasattr(llm, "base_model"):
            # base_model may be a wrapper; common case is .base_model.model
            bm = llm.base_model
            return getattr(bm, "model", bm)
    return llm


def save_ckpt(path, model, optimizer=None, step=None, best_val=None):
    """
    Saves:
      - PEFT adapters from model.llm (must be a PeftModel)
      - projector weights (projector.pt)
      - trainer_state.pt
    """
    os.makedirs(path, exist_ok=True)

    if not isinstance(model.llm, PeftModel):
        raise ValueError(
            "save_ckpt() expected model.llm to be a PEFT PeftModel (LoRA/QLoRA). "
            "Right now model.llm is not PEFT-wrapped. Make sure you built the LLM with get_peft_model()."
        )

    # 1) save adapters
    model.llm.save_pretrained(path)

    # 2) save projector (store on CPU to avoid GPU map issues)
    proj_sd = {k: v.detach().cpu() for k, v in model.projector.state_dict().items()}
    torch.save(proj_sd, os.path.join(path, "projector.pt"))

    # 3) trainer state
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
    Loads:
      - PEFT adapters into model.llm
      - projector weights
      - trainer_state.pt (optional)
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist or is not a directory: {path}")

    # 1) attach adapters to BASE model
    base_llm = _unwrap_base_llm(model.llm)
    model.llm = PeftModel.from_pretrained(
        base_llm,
        path,
        is_trainable=(optimizer is not None),
    )

    # 2) load projector
    proj_path = os.path.join(path, "projector.pt")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Missing projector.pt in {path}")

    sd = torch.load(proj_path, map_location=map_location)
    model.projector.load_state_dict(sd)

    # move projector to LLM device (important!)
    llm_device = model.llm.get_input_embeddings().weight.device
    model.projector.to(llm_device)

    # 3) trainer state
    state_path = os.path.join(path, "trainer_state.pt")
    state = None
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=map_location)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])

    return model, state
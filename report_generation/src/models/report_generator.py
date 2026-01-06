# src/models/report_generator.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .projector import MLPProjector


class VisionLLMReportGenerator(nn.Module):
    """
    Vision encoder (frozen) -> projector -> LLM (optionally QLoRA/PEFT)
    """

    def __init__(
        self,
        vision_encoder,
        llm_name: str,
        use_qlora: bool = False,
        qlora_kwargs: dict | None = None,
    ):
        super().__init__()
        self.vision = vision_encoder  # frozen outside
        qlora_kwargs = qlora_kwargs or {}

        # ---- LLM + tokenizer ----
        if use_qlora:
            # expects: (llm, tokenizer)
            from src.models.qlora import build_vicuna_qlora

            self.llm, self.tokenizer = build_vicuna_qlora(llm_name, **qlora_kwargs)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm.config.use_cache = False  # training

        # ---- projector ----
        vision_dim = getattr(self.vision, "embed_dim", None)
        if vision_dim is None:
            raise ValueError("vision_encoder must define .embed_dim")

        llm_hidden = self.llm.config.hidden_size
        self.projector = MLPProjector(in_dim=vision_dim, out_dim=llm_hidden)

        self._cached_llm_device = None

    def freeze_vision(self):
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

    def _llm_device(self) -> torch.device:
        """
        When using device_map="auto" / quant, params can be sharded.
        Most reliable single device to target: input embedding weight device.
        """
        if self._cached_llm_device is None:
            emb = self.llm.get_input_embeddings()
            self._cached_llm_device = emb.weight.device
        return self._cached_llm_device

    def _vision_device(self) -> torch.device:
        # vision is not sharded; safe to take first param/buffer device
        for p in self.vision.parameters():
            return p.device
        for b in self.vision.buffers():
            return b.device
        return torch.device("cpu")

    def _ensure_projector_on_llm_device(self):
        llm_device = self._llm_device()
        if next(self.projector.parameters()).device != llm_device:
            self.projector.to(llm_device)

    def forward(self, batch):
        """
        batch:
          images [B,3,H,W] float in [0,1]
          input_ids [B,T]
          attention_mask [B,T]
          labels [B,T]
        """
        llm_device = self._llm_device()
        vision_device = self._vision_device()
        self._ensure_projector_on_llm_device()

        # images -> vision device
        images = batch["images"].to(vision_device, non_blocking=True)

        # text -> llm device
        input_ids = batch["input_ids"].to(llm_device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(llm_device, non_blocking=True)
        labels = batch["labels"].to(llm_device, non_blocking=True)

        # 1) vision tokens (frozen)
        with torch.no_grad():
            patch_tokens = self.vision(images)  # [B,N,D] on vision_device

        # move tokens to llm device for projection + concat
        patch_tokens = patch_tokens.to(llm_device, non_blocking=True)

        # 2) project to LLM hidden
        image_embeds = self.projector(patch_tokens)  # [B,N,H]

        # 3) text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,T,H]

        # 4) concat
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # [B,N+T,H]

        # 5) attention
        B, N, _ = image_embeds.shape
        img_attn = torch.ones((B, N), dtype=attention_mask.dtype, device=llm_device)
        attn = torch.cat([img_attn, attention_mask], dim=1)  # [B,N+T]

        # 6) labels (ignore image tokens)
        img_labels = torch.full((B, N), -100, dtype=labels.dtype, device=llm_device)
        full_labels = torch.cat([img_labels, labels], dim=1)  # [B,N+T]

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=full_labels)
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        images,
        input_ids,
        attention_mask=None,
        max_new_tokens=150,
        temperature=0.7,
    ):
        llm_device = self._llm_device()
        vision_device = self._vision_device()
        self._ensure_projector_on_llm_device()

        images = images.to(vision_device, non_blocking=True)
        input_ids = input_ids.to(llm_device, non_blocking=True)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=llm_device)
        else:
            attention_mask = attention_mask.to(llm_device, non_blocking=True)

        patch_tokens = self.vision(images)  # on vision_device
        patch_tokens = patch_tokens.to(llm_device, non_blocking=True)

        image_embeds = self.projector(patch_tokens)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        B, N, _ = image_embeds.shape
        img_attn = torch.ones((B, N), dtype=attention_mask.dtype, device=llm_device)
        attn = torch.cat([img_attn, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
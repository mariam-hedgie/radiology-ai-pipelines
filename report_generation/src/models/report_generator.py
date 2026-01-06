# src/models/report_generator.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .projector import MLPProjector

class VisionLLMReportGenerator(nn.Module):
    def __init__(self, vision_encoder, llm_name: str):
        super().__init__()
        self.vision = vision_encoder  # frozen
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm.config.use_cache = False  # training

        # projector: 768 -> LLM hidden
        llm_hidden = self.llm.config.hidden_size
        vision_dim = getattr(vision_encoder, "embed_dim", None)
        if vision_dim is None:
            raise ValueError("vision_encoder must define .embed_dim")
        self.projector = MLPProjector(in_dim=vision_dim, out_dim=llm_hidden)

    def freeze_vision(self):
        self.vision.eval()
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, batch, tokenizer=None):
        """
        batch contains:
        images [B,3,H,W] in [0,1]
        input_ids [B,T]
        attention_mask [B,T]
        labels [B,T]   (prompt masked as -100)
        """
        device = next(self.parameters()).device

        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 1) vision tokens (frozen)
        with torch.no_grad():
            patch_tokens = self.vision(images)              # [B,N,D]

        # 2) project to LLM hidden
        image_embeds = self.projector(patch_tokens)         # [B,N,H]

        # 3) text token embeddings for the whole sequence
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,T,H]

        # 4) prepend image embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # [B,N+T,H]

        # 5) expand attention mask (image tokens are all "real")
        B, N, _ = image_embeds.shape
        img_attn = torch.ones((B, N), dtype=attention_mask.dtype, device=device)
        attn = torch.cat([img_attn, attention_mask], dim=1)            # [B,N+T]

        # 6) expand labels: ignore image tokens
        img_labels = torch.full((B, N), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([img_labels, labels], dim=1)           # [B,N+T]

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=full_labels
        )
        return out.loss

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask=None, max_new_tokens=150, temperature=0.7):
        device = next(self.parameters()).device
        images = images.to(device)
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        patch_tokens = self.vision(images)                 # [B,N,D]
        image_embeds = self.projector(patch_tokens)        # [B,N,H]
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        B, N, _ = image_embeds.shape
        img_attn = torch.ones((B, N), dtype=attention_mask.dtype, device=device)
        attn = torch.cat([img_attn, attention_mask], dim=1)

        gen = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        return gen
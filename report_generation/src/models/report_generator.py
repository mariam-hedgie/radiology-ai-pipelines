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
            # fallback: run one dummy forward to infer D
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)  # or cfg.image_size if you pass it in
                pt = vision_encoder(dummy)
                vision_dim = pt.shape[-1]

        self.projector = MLPProjector(in_dim=vision_dim, out_dim=llm_hidden)

    def freeze_vision(self):
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, batch, tokenizer):
        """
        batch contains:
          images [B,3,H,W]
          prompt_input_ids [B,Tp]
          target_input_ids [B,Tt]
        We build inputs_embeds = [image_embeds, prompt_embeds, target_embeds]
        labels = [-100 for image+prompt, target_ids for target]
        """
        images = batch["images"].to(next(self.parameters()).device)
        prompt_ids = batch["prompt_input_ids"].to(images.device)
        target_ids = batch["target_input_ids"].to(images.device)

        # 1) vision tokens (frozen)
        with torch.no_grad():
            patch_tokens = self.vision(images)           # [B,N,768]

        # 2) project to LLM space
        image_embeds = self.projector(patch_tokens)      # [B,N,H]

        # 3) text embeddings
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # [B,Tp,H]
        target_embeds = self.llm.get_input_embeddings()(target_ids)  # [B,Tt,H]

        # 4) concat
        inputs_embeds = torch.cat([image_embeds, prompt_embeds, target_embeds], dim=1)

        # 5) attention mask: 1s for everything
        B, N, _ = image_embeds.shape
        Tp = prompt_ids.shape[1]
        Tt = target_ids.shape[1]
        attn_mask = torch.ones((B, N + Tp + Tt), dtype=torch.long, device=images.device)

        # 6) labels: ignore image+prompt, supervise target
        ignore = torch.full((B, N + Tp), -100, dtype=torch.long, device=images.device)
        labels = torch.cat([ignore, target_ids], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, images, prompt_ids, max_new_tokens=150, temperature=0.7):
        device = next(self.parameters()).device
        images = images.to(device)
        prompt_ids = prompt_ids.to(device)

        patch_tokens = self.vision(images)                 # [B,N,768]
        image_embeds = self.projector(patch_tokens)        # [B,N,H]
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)

        inputs_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)

        B, N, _ = image_embeds.shape
        Tp = prompt_ids.shape[1]
        attn_mask = torch.ones((B, N + Tp), dtype=torch.long, device=device)

        gen = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        return gen
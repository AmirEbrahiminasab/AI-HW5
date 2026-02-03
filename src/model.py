# model.py
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from src.utils import mean_pooling, l2_normalize


@dataclass
class ModelConfig:
    text_model_name: str = "microsoft/MiniLM-L12-H384-uncased"
    embed_dim: int = 768
    dropout: float = 0.1
    init_temperature: float = 0.07  # CLIP-like


class TextAligner(nn.Module):
    """
    Trainable text encoder that maps text -> embed_dim (e.g., 512)
    and aligns with frozen image embeddings via CLIP-style contrastive loss.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_backbone = AutoModel.from_pretrained(cfg.text_model_name, use_safetensors=True)

        hidden = self.text_backbone.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(hidden, cfg.embed_dim)

        # CLIP uses a learned logit_scale; initialize from temperature.
        logit_scale = torch.log(torch.tensor(1.0 / cfg.init_temperature))
        self.logit_scale = nn.Parameter(logit_scale)

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.text_backbone(**text_inputs)
        pooled = mean_pooling(out.last_hidden_state, text_inputs["attention_mask"])
        pooled = self.dropout(pooled)
        z = self.proj(pooled)
        z = l2_normalize(z)
        return z

    def forward(self, text_inputs: Dict[str, torch.Tensor], image_embeds: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        image_embeds: [B, D] (already extracted by frozen image encoder, ideally L2-normalized)
        """
        text_embeds = self.encode_text(text_inputs)          # [B, D]
        image_embeds = l2_normalize(image_embeds)            # safety normalize

        # Clamp as done in many CLIP impls for stability
        logit_scale = self.logit_scale.clamp(max=torch.log(torch.tensor(100.0, device=self.logit_scale.device)))
        scale = logit_scale.exp()

        logits = scale * (text_embeds @ image_embeds.t())    # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_t2i = nn.functional.cross_entropy(logits, labels)
        loss_i2t = nn.functional.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_t2i + loss_i2t)

        with torch.no_grad():
            acc1 = (logits.argmax(dim=-1) == labels).float().mean().item()

        metrics = {
            "loss": float(loss.item()),
            "acc_inbatch@1": float(acc1),
            "logit_scale": float(scale.item()),
        }
        return loss, metrics

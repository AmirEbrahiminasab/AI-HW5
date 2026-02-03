# data.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from src.hf_compat import load_dataset_compat


def load_image_text_dataset(dataset_name: str, split: str):
    """
    Default expects datasets like 'flickr30k' that return fields:
      - 'image' (PIL)
      - 'caption' (list[str] or str depending on dataset)
    """
    ds = load_dataset_compat(dataset_name, split)
    return ds


class CaptionPairsDataset(Dataset):
    """
    Produces (caption, image_embed) pairs.
    We assume image embeddings were precomputed in the same order as ds items:
      embeds[image_idx] corresponds to ds[image_idx]["image"].

    If ds has multiple captions per image, we flatten them to multiple pairs.
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_model_name: str,
        image_embeds_path: str,
        max_length: int = 32,
    ):
        super().__init__()
        self.ds = load_image_text_dataset(dataset_name, split)
        self.dataset_name = dataset_name
        self.split = split

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.max_length = max_length

        self.image_embeds_path = image_embeds_path
        self._embeds = None  # lazy-mmap per worker

        # Build flattened (image_idx, caption)
        self.pairs: List[Tuple[int, str]] = []
        for i in range(len(self.ds)):
            cap = self.ds[i].get("caption", None)
            if cap is None:
                raise ValueError(f"Expected 'caption' field in dataset {dataset_name}. Got keys: {list(self.ds[i].keys())}")

            if isinstance(cap, (list, tuple)):
                for c in cap:
                    if isinstance(c, str) and c.strip():
                        self.pairs.append((i, c.strip()))
            elif isinstance(cap, str):
                if cap.strip():
                    self.pairs.append((i, cap.strip()))
            else:
                raise ValueError(f"Unsupported caption type: {type(cap)}")

        if len(self.pairs) == 0:
            raise ValueError("No (image, caption) pairs found after flattening captions.")

    def _get_embeds(self) -> np.ndarray:
        if self._embeds is None:
            self._embeds = np.load(self.image_embeds_path, mmap_mode="r")  # [N, D], float16/float32
        return self._embeds

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_idx, caption = self.pairs[idx]
        embeds = self._get_embeds()
        img_emb = embeds[image_idx]  # numpy row [D]

        enc = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # squeeze batch dim from tokenizer
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image_emb": torch.from_numpy(np.array(img_emb, copy=True)),
        }
        return item


def collate_caption_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pad text
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    image_emb = torch.stack([b["image_emb"] for b in batch], dim=0)
    return {
        "text_inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
        "image_emb": image_emb,
    }

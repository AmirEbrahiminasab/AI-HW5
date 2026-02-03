# query.py
import argparse
import os

import faiss
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image

from src.model import TextAligner, ModelConfig
from src.utils import l2_normalize, load_json
from src.hf_compat import load_dataset_compat

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="flickr30k")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--image_embeds", type=str, required=True)

    ap.add_argument("--checkpoint", type=str, required=True, help="Path to text_aligner.pt")
    ap.add_argument("--model_cfg", type=str, required=True, help="Path to model_cfg.json saved during training")

    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--save_dir", type=str, default="retrieval_results")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load index
    index = faiss.read_index(args.index_path)

    # Load image embeds (same order as dataset split)
    embeds = np.load(args.image_embeds, mmap_mode="r")  # [N, D]
    xb = np.asarray(embeds, dtype=np.float32)
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)

    # Load model config and model
    cfg_dict = load_json(args.model_cfg)
    cfg = ModelConfig(**cfg_dict)
    model = TextAligner(cfg).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, use_fast=True)

    enc = tokenizer(args.query, return_tensors="pt", truncation=True, max_length=32)
    enc = {k: v.to(device) for k, v in enc.items()}
    text_z = model.encode_text(enc)  # [1, D]
    text_z = l2_normalize(text_z).detach().cpu().numpy().astype(np.float32)

    # Search
    scores, idxs = index.search(text_z, args.topk)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    print("\nQuery:", args.query)
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        print(f"{rank:02d}. image_idx={i}  score={s:.4f}")

    # Optional: render retrieved images
    ds = load_dataset_compat(args.dataset, split=args.split)
    for rank, i in enumerate(idxs, start=1):
        img = ds[i]["image"].convert("RGB")
        out_path = os.path.join(args.save_dir, f"rank_{rank:02d}_idx_{i}.jpg")
        img.save(out_path)

    print(f"\nSaved top-{args.topk} images to: {args.save_dir}")


if __name__ == "__main__":
    main()

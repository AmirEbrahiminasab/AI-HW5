# vision_cache.py
import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from src.utils import set_seed
from src.hf_compat import load_dataset_compat

def collate_images(batch):
    images = [x["image"].convert("RGB") for x in batch]
    return {"images": images}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="flickr30k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--image_model", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Full Model: {args.image_model}...")

    # ---------------------------------------------------------
    # FIX: Use CLIPModel (Parent) instead of CLIPVisionModel
    # This automatically handles the config hierarchy and dimensions.
    # ---------------------------------------------------------
    try:
        model = CLIPModel.from_pretrained(args.image_model).to(device)
    except Exception:
        # Fallback for safetensors vs bin issues
        model = CLIPModel.from_pretrained(args.image_model, use_safetensors=False).to(device)

    processor = CLIPProcessor.from_pretrained(args.image_model)
    
    model.eval()
    if args.fp16 and device == "cuda":
        model.half()

    ds = load_dataset_compat(args.dataset, split=args.split)
    
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=collate_images,
        pin_memory=True
    )

    all_embeds: List[np.ndarray] = []
    
    print(f"Processing {len(ds)} images...")
    for batch in tqdm(dl):
        inputs = processor(images=batch["images"], return_tensors="pt", padding=True)
        # remove text inputs if processor added them by mistake
        if "input_ids" in inputs: del inputs["input_ids"]
        if "attention_mask" in inputs: del inputs["attention_mask"]
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if args.fp16 and device == "cuda":
             inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            # This is the standard API, handles projection automatically
            embeds = model.get_image_features(**inputs)
            embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-12)
            
        all_embeds.append(embeds.cpu().float().numpy())

    arr = np.concatenate(all_embeds, axis=0).astype(np.float16)
    
    ds_key = args.dataset.replace("/", "_")
    fname = f"{ds_key}_{args.split}_image_embeds_fp16.npy"
    out_path = os.path.join(args.out_dir, fname)
    np.save(out_path, arr)
    print(f"✅ Saved {arr.shape} to {out_path}")

if __name__ == "__main__":
    main()
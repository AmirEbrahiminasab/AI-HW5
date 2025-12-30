# faiss_index.py
import argparse
import os

import numpy as np
import faiss
from tqdm import tqdm

from src.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_embeds", type=str, required=True, help="Path to cached image embeds .npy")
    ap.add_argument("--out_dir", type=str, default="faiss_out")
    ap.add_argument("--index_name", type=str, default="index_flatip.faiss")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    embeds = np.load(args.image_embeds, mmap_mode="r") 
    N, D = embeds.shape
    print(f"Loaded embeds: {args.image_embeds}  shape={embeds.shape} dtype={embeds.dtype}")

    xb = np.asarray(embeds, dtype=np.float32)

    norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
    xb = xb / norms

    index = faiss.IndexFlatIP(D)
    index.add(xb)
    out_path = os.path.join(args.out_dir, args.index_name)
    faiss.write_index(index, out_path)

    meta = {
        "image_embeds": args.image_embeds,
        "num_vectors": int(N),
        "dim": int(D),
        "faiss_type": "IndexFlatIP (cosine via normalized vectors)",
    }
    save_json(os.path.join(args.out_dir, "index_meta.json"), meta)

    print(f"Saved FAISS index: {out_path}")


if __name__ == "__main__":
    main()

# vision_cache.py
# TODO: this file is for caching the embeddings of the image encoder, as image encoder is freezed through our training.
# I suggest to build it based on the arguments passed to the script based on the backend.py
import argparse
import os

from src.utils import set_seed


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

if __name__ == "__main__":
    main()

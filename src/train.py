# train.py
# TODO: this file is for training the text encoder alignment model.
# I suggest to build it based on the arguments passed to the script based on the backend.py
import argparse
import os

from src.utils import set_seed, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="flickr30k")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="validation")
    ap.add_argument("--train_image_embeds", type=str, required=True)
    ap.add_argument("--val_image_embeds", type=str, required=True)

    ap.add_argument("--text_model", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    ap.add_argument("--embed_dim", type=int, default=768)
    ap.add_argument("--max_length", type=int, default=64)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--out_dir", type=str, default="runs/text_aligner")
    ap.add_argument("--eval_every_steps", type=int, default=500)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    

if __name__ == "__main__":
    main()

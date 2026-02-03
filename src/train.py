# train.py
import argparse
import os
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.data_loader import CaptionPairsDataset, collate_caption_pairs
from src.model import TextAligner, ModelConfig
from src.utils import set_seed, save_json


@torch.no_grad()
def eval_inbatch_recall(model: TextAligner, dl, accelerator: Accelerator, k_list=(1, 5, 10), max_batches: int = 100) -> Dict[str, float]:
    model.eval()
    totals = {f"R@{k}": 0.0 for k in k_list}
    n = 0

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break

        text_inputs = {k: v for k, v in batch["text_inputs"].items()}
        image_emb = batch["image_emb"]

        text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
        image_emb = image_emb.to(accelerator.device)

        text_z = model.encode_text(text_inputs)   # [B, D], normalized
        img_z = image_emb / (image_emb.norm(dim=-1, keepdim=True) + 1e-12)

        text_z = text_z.float()
        img_z = img_z.float()
        sim = text_z @ img_z.t()
        ranks = sim.argsort(dim=-1, descending=True)  # [B, B]
        gt = torch.arange(sim.size(0), device=sim.device).unsqueeze(-1)  # [B, 1]

        for k in k_list:
            hit = (ranks[:, :k] == gt).any(dim=-1).float().mean().item()
            totals[f"R@{k}"] += hit

        n += 1

    if n == 0:
        return {**totals}

    return {k: v / n for k, v in totals.items()}


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

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    accelerator.print(f"Using device={accelerator.device}, mixed_precision={accelerator.mixed_precision}")

    train_ds = CaptionPairsDataset(
        dataset_name=args.dataset,
        split=args.train_split,
        text_model_name=args.text_model,
        image_embeds_path=args.train_image_embeds,
        max_length=args.max_length,
    )
    val_ds = CaptionPairsDataset(
        dataset_name=args.dataset,
        split=args.val_split,
        text_model_name=args.text_model,
        image_embeds_path=args.val_image_embeds,
        max_length=args.max_length,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_caption_pairs,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, 512),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_caption_pairs,
        pin_memory=True,
    )

    cfg = ModelConfig(text_model_name=args.text_model, embed_dim=args.embed_dim, dropout=args.dropout)
    model = TextAligner(cfg)

    # Only text side is trainable (vision is already cached/frozen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_dl) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

    # Save config
    if accelerator.is_main_process:
        save_json(os.path.join(args.out_dir, "train_args.json"), vars(args))
        save_json(os.path.join(args.out_dir, "model_cfg.json"), asdict(cfg))

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            text_inputs = {k: v.to(accelerator.device) for k, v in batch["text_inputs"].items()}
            image_emb = batch["image_emb"].to(accelerator.device)

            with accelerator.accumulate(model):
                loss, metrics = model(text_inputs, image_emb)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_local_main_process:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            if (global_step % args.eval_every_steps) == 0:
                rec = eval_inbatch_recall(model, val_dl, accelerator, max_batches=50)
                accelerator.print(f"[step {global_step}] val " + " ".join([f"{k}={v:.3f}" for k, v in rec.items()]))

                # checkpoint
                if accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.out_dir, "checkpoints", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, "text_aligner.pt"))

        # end epoch checkpoint
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(args.out_dir, "checkpoints", f"epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, "text_aligner.pt"))

    accelerator.print("Done.")


if __name__ == "__main__":
    main()

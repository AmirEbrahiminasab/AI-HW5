import argparse
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import faiss
from flask import Flask, request, render_template, url_for

from transformers import AutoTokenizer
from datasets import load_dataset

from src.model import TextAligner, ModelConfig
from src.utils import load_json, l2_normalize
from src.hf_compat import load_dataset_compat

# ----------------------------
# Helpers
# ----------------------------

def run_cmd(cmd: List[str]) -> None:
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def guess_latest_checkpoint(run_dir: str) -> Optional[str]:
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None

    # Prefer epoch_*; else step_*
    candidates = []
    for name in os.listdir(ckpt_root):
        m_epoch = re.match(r"epoch_(\d+)$", name)
        m_step = re.match(r"step_(\d+)$", name)
        if m_epoch or m_step:
            folder = os.path.join(ckpt_root, name)
            ckpt = os.path.join(folder, "text_aligner.pt")
            if os.path.isfile(ckpt):
                score = int(m_epoch.group(1)) * 10**12 if m_epoch else int(m_step.group(1))
                candidates.append((score, ckpt))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def clean_results_dir(path: str, keep_last: int = 200) -> None:
    """
    Prevent static/results from growing forever.
    Keeps most recent N files by mtime.
    """
    if not os.path.isdir(path):
        return
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if len(files) <= keep_last:
        return
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for f in files[keep_last:]:
        try:
            os.remove(f)
        except Exception:
            pass


# ----------------------------
# Pipeline: cache -> train -> index
# ----------------------------

@dataclass
class PipelinePaths:
    cache_dir: str
    run_dir: str
    faiss_dir: str

    train_embeds: str
    val_embeds: str
    db_embeds: str  # embeddings used for FAISS index

    faiss_index_path: str
    model_cfg_path: str


def make_paths(dataset: str, cache_dir: str, run_dir: str, faiss_dir: str, db_split: str) -> PipelinePaths:
    ds_key = dataset.replace("/", "_")
    train_embeds = os.path.join(cache_dir, f"{ds_key}_train_image_embeds_fp16.npy")
    val_embeds = os.path.join(cache_dir, f"{ds_key}_validation_image_embeds_fp16.npy")
    db_embeds = os.path.join(cache_dir, f"{ds_key}_{db_split}_image_embeds_fp16.npy")

    faiss_index_path = os.path.join(faiss_dir, "index_flatip.faiss")
    model_cfg_path = os.path.join(run_dir, "model_cfg.json")

    return PipelinePaths(
        cache_dir=cache_dir,
        run_dir=run_dir,
        faiss_dir=faiss_dir,
        train_embeds=train_embeds,
        val_embeds=val_embeds,
        db_embeds=db_embeds,
        faiss_index_path=faiss_index_path,
        model_cfg_path=model_cfg_path,
    )


def pipeline_run_all(
    dataset: str,
    image_model: str,
    text_model: str,
    cache_dir: str,
    run_dir: str,
    faiss_dir: str,
    db_split: str,
    batch_size_cache: int,
    batch_size_train: int,
    epochs: int,
    mixed_precision: str,
) -> PipelinePaths:
    ensure_dir(cache_dir)
    ensure_dir(run_dir)
    ensure_dir(faiss_dir)

    paths = make_paths(dataset, cache_dir, run_dir, faiss_dir, db_split)

    # 1) Cache image embeddings for train/validation/db_split
    run_cmd(["python3", "-m", "src.vision_cache", "--dataset", dataset, "--split", "train",
             "--image_model", image_model, "--out_dir", cache_dir, "--batch_size", str(batch_size_cache), "--fp16"])
    run_cmd(["python3", "-m", "src.vision_cache", "--dataset", dataset, "--split", "validation",
             "--image_model", image_model, "--out_dir", cache_dir, "--batch_size", str(batch_size_cache), "--fp16"])
    run_cmd(["python3", "-m", "src.vision_cache", "--dataset", dataset, "--split", db_split,
             "--image_model", image_model, "--out_dir", cache_dir, "--batch_size", str(batch_size_cache), "--fp16"])

    # 2) Train text encoder alignment
    run_cmd([
        "python3", "-m", "src.train",
        "--dataset", dataset,
        "--train_split", "train",
        "--val_split", "validation",
        "--train_image_embeds", paths.train_embeds,
        "--val_image_embeds", paths.val_embeds,
        "--text_model", text_model,
        "--out_dir", run_dir,
        "--batch_size", str(batch_size_train),
        "--epochs", str(epochs),
        "--mixed_precision", mixed_precision
    ])

    # 3) Build FAISS index for db_split
    run_cmd([
        "python3", "-m", "src.faiss_index",
        "--image_embeds", paths.db_embeds,
        "--out_dir", faiss_dir,
        "--index_name", "index_flatip.faiss"
    ])

    return paths


# ----------------------------
# Web App: "Gigoole"
# ----------------------------

def create_app(
    dataset: str,
    split: str,
    faiss_index_path: str,
    checkpoint_path: str,
    model_cfg_path: str,
    topk_default: int = 8,
) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    ensure_dir(os.path.join(app.static_folder, "results"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FAISS index
    index = faiss.read_index(faiss_index_path)

    # Load dataset (same split used to build embeddings/index)
    ds = load_dataset_compat(dataset, split=split)

    # Load model cfg + model
    cfg_dict = load_json(model_cfg_path)
    cfg = ModelConfig(**cfg_dict)
    model = TextAligner(cfg).to(device)
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, use_fast=True)

    @torch.no_grad()
    def retrieve(query: str, topk: int) -> List[Tuple[str, float, int]]:
        enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=32)
        enc = {k: v.to(device) for k, v in enc.items()}
        text_z = model.encode_text(enc)  # [1, D]
        text_z = l2_normalize(text_z).detach().cpu().numpy().astype(np.float32)

        scores, idxs = index.search(text_z, topk)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        # Save images into static/results
        stamp = str(int(time.time() * 1000))
        out = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            img = ds[i]["image"].convert("RGB")
            fname = f"{stamp}_rank{rank:02d}_idx{i}.jpg"
            fpath = os.path.join(app.static_folder, "results", fname)
            img.save(fpath, quality=92)
            out.append((url_for("static", filename=f"results/{fname}"), float(s), int(i)))

        clean_results_dir(os.path.join(app.static_folder, "results"), keep_last=250)
        return out

    @app.get("/")
    def home():
        return render_template("index.html", topk=topk_default)

    @app.post("/search")
    def search():
        q = (request.form.get("q") or "").strip()
        try:
            topk = int(request.form.get("topk") or topk_default)
        except Exception:
            topk = topk_default
        topk = max(1, min(topk, 50))

        if not q:
            return render_template("index.html", topk=topk, error="Please type a query.")

        results = retrieve(q, topk)
        return render_template("results.html", query=q, results=results, topk=topk)

    return app


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run_all: cache -> train -> index
    p_run = sub.add_parser("run_all", help="Run caching, training, and FAISS indexing step-by-step.")
    p_run.add_argument("--dataset", type=str, default="flickr30k")
    p_run.add_argument("--image_model", type=str, default="openai/clip-vit-base-patch32")
    p_run.add_argument("--text_model", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    p_run.add_argument("--cache_dir", type=str, default="cache")
    p_run.add_argument("--run_dir", type=str, default="runs/flickr30k_minilm")
    p_run.add_argument("--faiss_dir", type=str, default="faiss_db")
    p_run.add_argument("--db_split", type=str, default="test")
    p_run.add_argument("--batch_size_cache", type=int, default=128)
    p_run.add_argument("--batch_size_train", type=int, default=256)
    p_run.add_argument("--epochs", type=int, default=5)
    p_run.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # serve: start web server
    p_srv = sub.add_parser("serve", help="Start the Gigoole web app on localhost.")
    p_srv.add_argument("--dataset", type=str, default="flickr30k")
    p_srv.add_argument("--split", type=str, default="test", help="Must match the split used to build the FAISS index.")
    p_srv.add_argument("--faiss_index", type=str, default="faiss_db/index_flatip.faiss")
    p_srv.add_argument("--run_dir", type=str, default="runs/flickr30k_minilm")
    p_srv.add_argument("--checkpoint", type=str, default="", help="Path to text_aligner.pt (if empty, auto-pick latest).")
    p_srv.add_argument("--host", type=str, default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=5000)
    p_srv.add_argument("--topk", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "run_all":
        paths = pipeline_run_all(
            dataset=args.dataset,
            image_model=args.image_model,
            text_model=args.text_model,
            cache_dir=args.cache_dir,
            run_dir=args.run_dir,
            faiss_dir=args.faiss_dir,
            db_split=args.db_split,
            batch_size_cache=args.batch_size_cache,
            batch_size_train=args.batch_size_train,
            epochs=args.epochs,
            mixed_precision=args.mixed_precision,
        )
        print("\n✅ Pipeline complete.")
        print("Model cfg:", paths.model_cfg_path)
        print("FAISS index:", paths.faiss_index_path)
        ckpt = guess_latest_checkpoint(args.run_dir)
        print("Latest checkpoint guess:", ckpt)

    elif args.cmd == "serve":
        model_cfg_path = os.path.join(args.run_dir, "model_cfg.json")
        if not os.path.isfile(model_cfg_path):
            raise FileNotFoundError(f"Missing model config: {model_cfg_path}. Did you run training?")

        ckpt = args.checkpoint.strip() or guess_latest_checkpoint(args.run_dir)
        if not ckpt or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                "Could not find a checkpoint. Provide --checkpoint or run training to generate checkpoints."
            )

        if not os.path.isfile(args.faiss_index):
            raise FileNotFoundError(f"Missing FAISS index: {args.faiss_index}. Build it first.")

        app = create_app(
            dataset=args.dataset,
            split=args.split,
            faiss_index_path=args.faiss_index,
            checkpoint_path=ckpt,
            model_cfg_path=model_cfg_path,
            topk_default=args.topk,
        )
        print(f"\n🚀 Gigoole running at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

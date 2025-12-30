#!/usr/bin/env bash
set -e

DATASET="${1:-nlphuji/flickr30k}"
IMAGE_MODEL="${2:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"
TEXT_MODEL="${3:-sentence-transformers/all-mpnet-base-v2}"
DB_SPLIT="${4:-test}"

CACHE_DIR="cache"
RUN_DIR="runs/flickr30k_minilm"
FAISS_DIR="faiss_db"

python3 -m src.backend run_all \
  --dataset "$DATASET" \
  --image_model "$IMAGE_MODEL" \
  --text_model "$TEXT_MODEL" \
  --db_split "$DB_SPLIT" \
  --cache_dir "$CACHE_DIR" \
  --run_dir "$RUN_DIR" \
  --faiss_dir "$FAISS_DIR"

python3 -m src.backend serve \
  --dataset "$DATASET" \
  --split "$DB_SPLIT" \
  --faiss_index "$FAISS_DIR/index_flatip.faiss" \
  --run_dir "$RUN_DIR" \
  --host 127.0.0.1 --port 5000 --topk 8

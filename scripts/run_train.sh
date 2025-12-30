
#!/usr/bin/env bash
set -e

# Here you can set the dataset, this is the dataset, I have chosen, you may choose another one however I don't recommend!
DATASET="${1:-nlphuji/flickr30k}"
# Here is the image encoder, you may choose any variation you like based on your computation power
IMAGE_MODEL="${2:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"
# Here is the text encoder, you may choose any variation you like based on your computation power
TEXT_MODEL="${3:-sentence-transformers/all-mpnet-base-v2}"
# The split used to build the database has to be test split that the models haven't seen in training.
DB_SPLIT="${4:-test}"
# Choose the following directories based on where you saved corresponding directories.
CACHE_DIR="cache"
RUN_DIR="runs/flickr30k_minilm"
FAISS_DIR="faiss_db"

# Here is the script that runs training.
python3 -m src.backend run_all \
  --dataset "$DATASET" \
  --image_model "$IMAGE_MODEL" \
  --text_model "$TEXT_MODEL" \
  --db_split "$DB_SPLIT" \
  --cache_dir "$CACHE_DIR" \
  --run_dir "$RUN_DIR" \
  --faiss_dir "$FAISS_DIR"


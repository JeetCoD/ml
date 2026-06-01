"""
config.py

all project-wide settings live here.
change a value once here and it updates everywhere automatically.
"""

from pathlib import Path

# -----------------------------------------------------------------
# folder paths
# -----------------------------------------------------------------

# the root of the project (two levels up from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# where the three processed play json files live
# NOTE: the dataset is in shakespeare_slm_dataset/ in the repo.
# you can either copy the files to data/processed/ OR just point
# DATA_DIR straight at shakespeare_slm_dataset/ as we do here.
DATA_DIR    = PROJECT_ROOT / "shakespeare_slm_dataset"
PROMPT_DIR  = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# -----------------------------------------------------------------
# the three compulsory play files
# -----------------------------------------------------------------

PLAY_FILES = {
    "hamlet":           DATA_DIR / "hamlet.json",
    "macbeth":          DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

# -----------------------------------------------------------------
# retrieval settings
# -----------------------------------------------------------------

# number of chunks to return for each query
DEFAULT_TOP_K = 3

# -----------------------------------------------------------------
# embedding model
# -----------------------------------------------------------------

# lightweight model that runs on cpu, no gpu needed.
# produces 384-dimensional vectors.
# ~90mb download on first use, then cached locally.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------------------
# cached index files
# -----------------------------------------------------------------

# save embeddings to disk so we only compute them once.
EMBEDDINGS_CACHE_PATH = RESULTS_DIR / "embeddings.npy"
CHUNKS_CACHE_PATH     = RESULTS_DIR / "chunks.json"

# -----------------------------------------------------------------
# language model
# -----------------------------------------------------------------

# qwen2.5:1.5b: a 1.5b parameter slm served by ollama.
# runs fully local on cpu, no api key, no internet at inference time.
# we picked this over flan-t5-base because qwen is newer (2024),
# instruction-tuned for chat, and produces longer grounded answers
# from a rag prompt. small enough to fit the slm framing in the spec.
LM_MODEL_NAME     = "qwen2.5:1.5b"
LM_MAX_NEW_TOKENS = 400
LM_TEMPERATURE    = 0.3  # low temperature for more grounded, less random answers

# -----------------------------------------------------------------
# chunking strategy
# -----------------------------------------------------------------

# "scene" = group all utterances in a scene into one chunk.
# this preserves narrative context and helps retrieval.
CHUNK_STRATEGY = "scene"

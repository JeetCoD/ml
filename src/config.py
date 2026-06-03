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

DATA_DIR    = PROJECT_ROOT / "shakespeare_slm_dataset"  # where the raw json play files live
PROMPT_DIR  = PROJECT_ROOT / "prompts"                  # folder that holds the system prompt text files
RESULTS_DIR = PROJECT_ROOT / "results"                  # where we write csvs, charts, and cached indexes

# -----------------------------------------------------------------
# the three compulsory play files
# -----------------------------------------------------------------

# each key is a short play name, each value is the path to its json file
PLAY_FILES = {
    "hamlet":           DATA_DIR / "hamlet.json",
    "macbeth":          DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

# -----------------------------------------------------------------
# retrieval settings
# -----------------------------------------------------------------

# number of chunks to return for each query so the llm has enough context
DEFAULT_TOP_K = 5

# -----------------------------------------------------------------
# embedding model
# -----------------------------------------------------------------

# lightweight model that runs on cpu, no gpu needed.
# all-minilm-l6-v2 is fast and small (~90mb) while still producing decent embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------------------
# cached index files
# -----------------------------------------------------------------

# save embeddings to disk so we only compute them once.
# next time the chatbot starts it loads these files instead of re-encoding everything.
EMBEDDINGS_CACHE_PATH = RESULTS_DIR / "embeddings.npy"
CHUNKS_CACHE_PATH     = RESULTS_DIR / "chunks.json"

# -----------------------------------------------------------------
# language model
# -----------------------------------------------------------------

# qwen2.5:1.5b: a 1.5 billion parameter small language model served by ollama.
# runs fully local on cpu, no api key, no internet needed at inference time.
# we picked this because it is instruction-tuned for chat and produces
# grounded answers from a rag prompt, and it is small enough to fit the slm framing.
LM_MODEL_NAME     = "qwen2.5:1.5b"
LM_MAX_NEW_TOKENS = 400   # max tokens the model generates per response
LM_TEMPERATURE    = 0.3   # low temperature so answers are more grounded and less random

# -----------------------------------------------------------------
# chunking strategy
# -----------------------------------------------------------------

# "scene" = group all utterances in a scene into one chunk.
# this preserves narrative context which helps retrieval find the right passage.
CHUNK_STRATEGY = "scene"

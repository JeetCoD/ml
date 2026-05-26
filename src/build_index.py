"""
build_index.py

this script loads all three shakespeare plays, creates scene-level chunks,
generates embeddings for each chunk, and saves everything to disk.

run this once before running the chatbot or the evaluator.
after it finishes, the chatbot loads the cached embeddings instead of
recomputing them, which saves a lot of time on every run.

output files:
  results/embeddings.npy   -> the embedding matrix (numpy array)
  results/chunks.json      -> the list of chunks with metadata

usage:
  python src/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# add src/ to the python path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_CACHE_PATH,
    CHUNKS_CACHE_PATH,
)
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def main() -> None:
    """
    full pipeline: load data, chunk it, embed it, save to disk, and test retrieval.
    """
    print("=" * 60)
    print("building shakespeare retrieval index")
    print("=" * 60)

    # step 1: load all play records from the json files
    print("\nstep 1: loading play records...")
    records = load_all_plays()

    # step 2: turn the records into scene-level chunks
    print("\nstep 2: creating scene-level chunks...")
    chunks = create_chunks(records)
    print(f"created {len(chunks)} chunks from {len(records)} records")

    # step 3: load the embedding model and generate embeddings
    print("\nstep 3: building the embedding index...")
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    # step 4: save embeddings and chunks to disk so we can reload later
    print("\nstep 4: saving embeddings and chunks to disk...")
    retriever.save_to_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    # step 5: run a quick test query to confirm retrieval is working
    print("\nstep 5: running a test retrieval query...")
    test_query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(test_query, top_k=DEFAULT_TOP_K)

    print(f"\nquery: '{test_query}'")
    print(f"top {DEFAULT_TOP_K} results:\n")

    for rank, (chunk, score) in enumerate(results, start=1):
        print("=" * 60)
        print(f"rank {rank} | cosine similarity score: {score:.4f}")
        print(format_chunk_for_display(chunk))
        print()

    print("=" * 60)
    print("index built and saved successfully.")
    print("you can now run rag_chatbot.py without rebuilding.")


if __name__ == "__main__":
    main()

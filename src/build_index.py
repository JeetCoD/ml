"""
build_index.py

run this script once before starting the chatbot.
it loads all three play files, creates scene-level chunks, encodes them
into embedding vectors, and saves everything to disk.

after this script finishes, rag_chatbot.py can load the cached index
instead of re-encoding from scratch each time.

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


# run the full pipeline: load, chunk, embed, save, and test
def main() -> None:
    """
    run the full index-building pipeline in five steps.
    after this completes, the chatbot can start instantly without re-encoding.
    """
    print("=" * 60)
    print("building shakespeare retrieval index")
    print("=" * 60)

    # step 1: load all play records from the json files on disk
    print("\nstep 1: loading play records...")
    records = load_all_plays()

    # step 2: turn the scene records into chunks that the retriever can index
    print("\nstep 2: creating scene-level chunks...")
    chunks = create_chunks(records)
    print(f"created {len(chunks)} chunks from {len(records)} records")

    # step 3: load the embedding model and encode every chunk into a vector
    print("\nstep 3: building the embedding index...")
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)

    # step 4: write the embeddings array and chunk list to disk for future runs
    print("\nstep 4: saving embeddings and chunks to disk...")
    retriever.save_to_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    # step 5: run a quick test query to confirm the retriever works correctly
    print("\nstep 5: running a test retrieval query...")
    test_query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(test_query, top_k=DEFAULT_TOP_K)

    print(f"\nquery: '{test_query}'")
    print(f"top {DEFAULT_TOP_K} results:\n")

    # show each result with its rank and cosine similarity score
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

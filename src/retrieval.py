"""
retrieval.py

this file handles embedding generation and similarity-based retrieval.
it also supports saving and loading embeddings from disk so we do not
have to recompute them every time we run the chatbot.

how it works:
  1. we load the sentence-transformers model once
  2. we encode each chunk's text into a vector (embedding)
  3. when a user asks a question, we embed the question too
  4. we compare the question embedding against all chunk embeddings
     using cosine similarity
  5. we return the top-k most similar chunks

cosine similarity measures the angle between two vectors.
a score close to 1.0 means very similar, close to 0 means unrelated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# type alias for a chunk dictionary
Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """
    embeds chunks and retrieves the most relevant ones for a query.
    can save and load embeddings from disk to avoid recomputation.
    """

    def __init__(self, embedding_model_name: str):
        """
        load the sentence transformer model.
        the model is downloaded from huggingface on first use (~90 mb).
        after that it is cached locally so it loads instantly.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is not installed. "
                "install it with: pip install sentence-transformers"
            ) from error

        print(f"loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)

        # these will be filled when we build or load the index
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        encode all chunks and store the embeddings in memory.
        this can take a minute if there are many chunks.
        """
        if not chunks:
            raise ValueError("no chunks provided to build_index(). check your data.")

        self.chunks = chunks

        # get the text from each chunk to embed
        texts = [chunk["text"] for chunk in chunks]

        print(f"encoding {len(texts)} chunks... (this may take a minute on cpu)")
        self.embeddings = np.asarray(
            self.model.encode(texts, show_progress_bar=True, batch_size=32)
        )
        print(f"done. embedding matrix shape: {self.embeddings.shape}")

    def save_to_disk(self, embeddings_path: Path, chunks_path: Path) -> None:
        """
        save the embeddings array and chunk list to disk.
        this means we only have to compute embeddings once.
        """
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("index has not been built yet. call build_index() first.")

        # make sure the results folder exists
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # save embeddings as a numpy binary file (fast and compact)
        np.save(embeddings_path, self.embeddings)

        # save the chunks as json so we can read them back easily
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        print(f"saved embeddings to: {embeddings_path}")
        print(f"saved chunks to:     {chunks_path}")

    def load_from_disk(self, embeddings_path: Path, chunks_path: Path) -> bool:
        """
        try to load embeddings and chunks from disk.
        returns true if successful, false if the files do not exist.
        """
        if not embeddings_path.exists() or not chunks_path.exists():
            return False

        # load the numpy embeddings array
        self.embeddings = np.load(embeddings_path)

        # load the chunks list from json
        with chunks_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"loaded {len(self.chunks)} chunks and embeddings from cache.")
        print(f"embedding matrix shape: {self.embeddings.shape}")
        return True

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        find the top-k most relevant chunks for a given query.
        returns a list of (chunk, similarity_score) pairs.
        """
        if self.embeddings is None or not self.chunks:
            raise RuntimeError(
                "index has not been built. call build_index() or load_from_disk() first."
            )

        # embed the query using the same model as the chunks
        query_vector = np.asarray(self.model.encode([query]))

        # compute cosine similarity between the query and every chunk
        similarity_scores = cosine_similarity(query_vector, self.embeddings)[0]

        # sort by score in descending order and take the top-k
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # return pairs of (chunk, score)
        results = [
            (self.chunks[i], float(similarity_scores[i]))
            for i in top_indices
        ]
        return results

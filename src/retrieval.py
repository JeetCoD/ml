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
this is why it works well for semantic search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# sentence_transformers must be imported before numpy/sklearn to avoid a
# windows dll conflict: tensorflow (loaded by sentence_transformers) and
# numpy's openblas both register incompatible blas symbols. loading tf first
# prevents the access-violation crash (exit 0xc0000005) on windows.
try:
    from sentence_transformers import SentenceTransformer
except ImportError as _st_error:
    raise ImportError(
        "sentence-transformers is not installed. "
        "install it with: pip install sentence-transformers"
    ) from _st_error

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# type alias for a chunk dictionary so function signatures are easier to read
Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """
    loads an embedding model, encodes chunks into vectors, and retrieves the
    most relevant chunks for a given query using cosine similarity.

    it also supports saving and loading the index from disk to avoid recomputing
    embeddings every time the program starts.
    """

    # load the sentence transformer model from huggingface
    def __init__(self, embedding_model_name: str):
        """
        download and load the sentence transformer model.

        inputs:
          embedding_model_name - the huggingface model name string (e.g. all-minilm-l6-v2)

        the model is about 90mb and is cached locally after the first download,
        so subsequent loads are instant.
        """
        print(f"loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)

        # these will be filled when we call build_index() or load_from_disk()
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None  # 2d array: rows=chunks, cols=embedding dims

    # encode all chunks and store the resulting vectors in memory
    def build_index(self, chunks: List[Chunk]) -> None:
        """
        encode every chunk's text into a vector and store the matrix in memory.

        inputs:
          chunks - list of chunk dicts, each must have a "text" field

        this step is the main bottleneck. encoding ~250 chunks on cpu takes about a minute.
        that is why we save the result to disk afterward so we only do it once.
        """
        if not chunks:
            raise ValueError("no chunks provided to build_index(). check your data.")

        self.chunks = chunks

        # extract just the text from each chunk so we can pass a list of strings to encode()
        texts = [chunk["text"] for chunk in chunks]

        print(f"encoding {len(texts)} chunks... (this may take a minute on cpu)")
        # encode() converts each text string into a fixed-length vector (embedding).
        # batch_size=32 means we process 32 texts at a time to manage memory.
        self.embeddings = np.asarray(
            self.model.encode(texts, show_progress_bar=True, batch_size=32)
        )
        print(f"done. embedding matrix shape: {self.embeddings.shape}")

    # write the index to disk so we can skip re-encoding next time
    def save_to_disk(self, embeddings_path: Path, chunks_path: Path) -> None:
        """
        save the embeddings array and the chunk list to disk.

        inputs:
          embeddings_path - path to write the .npy file (numpy binary format)
          chunks_path     - path to write the chunks as a .json file

        after saving, the next startup can call load_from_disk() instead of build_index().
        """
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("index has not been built yet. call build_index() first.")

        # make sure the results folder exists before trying to write into it
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # .npy is a compact numpy binary format, so this is fast and small
        np.save(embeddings_path, self.embeddings)

        # save chunks as json because it is human-readable and easy to load back
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        print(f"saved embeddings to: {embeddings_path}")
        print(f"saved chunks to:     {chunks_path}")

    # load a previously built index from disk to skip the slow encoding step
    def load_from_disk(self, embeddings_path: Path, chunks_path: Path) -> bool:
        """
        try to load embeddings and chunks from cached disk files.

        inputs:
          embeddings_path - path to the .npy embeddings file
          chunks_path     - path to the chunks .json file

        outputs:
          true if both files exist and were loaded successfully, false otherwise.
          if false, the caller should build the index from scratch.
        """
        # if either file is missing we cannot load from cache, so return false
        if not embeddings_path.exists() or not chunks_path.exists():
            return False

        # load the numpy array back into memory from the binary file
        self.embeddings = np.load(embeddings_path)

        # load the chunk list back from json
        with chunks_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"loaded {len(self.chunks)} chunks and embeddings from cache.")
        print(f"embedding matrix shape: {self.embeddings.shape}")
        return True

    # find the most relevant chunks for a query using cosine similarity
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        find the top-k chunks that are most similar to the query.

        inputs:
          query - the user's question as a plain string
          top_k - how many chunks to return

        outputs:
          a list of (chunk, similarity_score) pairs, sorted from highest to lowest score.

        how this works:
          1. we embed the query into a vector using the same model as the chunks
          2. we compute cosine similarity between that vector and every chunk vector
          3. cosine similarity gives a score between 0 and 1 for each chunk
          4. we sort by score and return the top-k
        """
        if self.embeddings is None or not self.chunks:
            raise RuntimeError(
                "index has not been built. call build_index() or load_from_disk() first."
            )

        # encode the query into a vector so we can compare it against the chunks.
        # we wrap it in a list because encode() expects a batch, then we take index [0].
        query_vector = np.asarray(self.model.encode([query]))

        # cosine_similarity returns a 2d array, so [0] gives us the 1d scores array.
        # each value is how similar that chunk is to the query (higher = more relevant).
        similarity_scores = cosine_similarity(query_vector, self.embeddings)[0]

        # argsort returns indices from smallest to largest, so [::-1] reverses it to
        # largest first. then [:top_k] takes only the top-k indices we want.
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # pair each winning chunk with its score so the caller can display both
        results = [
            (self.chunks[i], float(similarity_scores[i]))
            for i in top_indices
        ]
        return results

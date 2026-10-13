"""Text to vectors, in two forms.

Dense: a few hundred floats from a neural model. Texts with similar meaning land close
together even when they share no words. Sparse: one weight per word that occurs. It matches
exact terms such as a course code or "JEE", which a dense vector can blur.

The knowledge base stores both and searches both.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Protocol

Sparse = tuple[list[int], list[float]]  # (word ids, weights)

WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def dense(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...

    def sparse(self, texts: list[str], *, query: bool = False) -> list[Sparse]: ...


class HashEmbedder:
    """Deterministic stand-in for tests and offline runs. It downloads nothing.

    The dense vector is a bag of words hashed into a fixed number of buckets, so it only knows
    word overlap and not meaning. Never use it to report retrieval quality.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def dense(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in WORD.findall(text.lower()):
                vector[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    def sparse(self, texts: list[str], *, query: bool = False) -> list[Sparse]:
        out = []
        for text in texts:
            counts: dict[int, float] = {}
            for word in WORD.findall(text.lower()):
                word_id = zlib.crc32(word.encode())
                counts[word_id] = counts.get(word_id, 0.0) + 1.0
            out.append((list(counts), list(counts.values())))
        return out


class FastEmbedEmbedder:
    """Real embeddings on the CPU through ONNX. The first use downloads the model files."""

    def __init__(
        self,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "Qdrant/bm25",
        cache_dir: str | None = None,
    ):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(dense_model, cache_dir=cache_dir)
        self._sparse = SparseTextEmbedding(sparse_model, cache_dir=cache_dir)
        self.dim = len(next(iter(self._dense.embed(["dimension probe"]))))

    def dense(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        # Retrieval models embed a question and a passage differently, hence the two calls.
        embed = self._dense.query_embed if query else self._dense.embed
        return [vector.tolist() for vector in embed(texts)]

    def sparse(self, texts: list[str], *, query: bool = False) -> list[Sparse]:
        embed = self._sparse.query_embed if query else self._sparse.embed
        return [(item.indices.tolist(), item.values.tolist()) for item in embed(texts)]

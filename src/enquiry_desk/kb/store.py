"""The knowledge base in Qdrant: one collection, a dense and a sparse vector on every chunk."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from enquiry_desk.kb.chunking import Chunk
from enquiry_desk.kb.embed import Embedder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source: str
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = "knowledge"):
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def ensure(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config={
                "dense": models.VectorParams(size=self._embedder.dim, distance=models.Distance.COSINE)
            },
            # The IDF modifier makes Qdrant weight rare words above common ones at query time,
            # using its own count of how many chunks hold each word. That is the IDF half of BM25.
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Every query filters on tenant. Without this index the filter scans every point.
        self._client.create_payload_index(self._collection, "tenant", models.PayloadSchemaType.KEYWORD)

    def replace(self, tenant: str, chunks: list[Chunk]) -> int:
        """Index a tenant's chunks, removing whatever that tenant had before."""
        self.ensure()
        self._client.delete(self._collection, points_selector=models.FilterSelector(filter=_tenant(tenant)))
        if not chunks:
            return 0
        texts = [chunk.embed_text for chunk in chunks]
        dense = self._embedder.dense(texts)
        sparse = self._embedder.sparse(texts)
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": dense[i],
                        "sparse": models.SparseVector(indices=sparse[i][0], values=sparse[i][1]),
                    },
                    payload={"tenant": chunk.tenant, "doc": chunk.doc, "source": chunk.source, "text": chunk.text},
                )
                for i, chunk in enumerate(chunks)
            ],
        )
        return len(chunks)

    def search(self, tenant: str, query: str, k: int = 4, candidates: int = 20, mode: str = "hybrid") -> list[Hit]:
        """Hybrid search: the top dense matches and the top sparse matches, fused.

        Each prefetch returns its own ranked list. Reciprocal rank fusion then scores a chunk
        by its rank in each list (1 / (60 + rank), summed), so a chunk both lists agree on rises
        to the top. Only ranks are used, which matters because a cosine similarity and a BM25
        score are on different scales and cannot be added.

        mode "dense" or "sparse" searches one vector alone. The agent always uses "hybrid".
        The other two exist so the evaluation can show what each half contributes.
        """
        only_tenant = _tenant(tenant)
        dense = self._embedder.dense([query], query=True)[0]
        indices, values = self._embedder.sparse([query], query=True)[0]
        sparse = models.SparseVector(indices=indices, values=values)
        if mode == "hybrid":
            result = self._client.query_points(
                self._collection,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=candidates, filter=only_tenant),
                    models.Prefetch(query=sparse, using="sparse", limit=candidates, filter=only_tenant),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        else:
            result = self._client.query_points(
                self._collection,
                query=dense if mode == "dense" else sparse,
                using=mode,
                query_filter=only_tenant,
                limit=k,
                with_payload=True,
            )
        return [
            Hit(chunk_id=str(point.id), source=point.payload["source"], text=point.payload["text"], score=point.score)
            for point in result.points
        ]

    def count(self, tenant: str) -> int:
        self.ensure()
        return self._client.count(self._collection, count_filter=_tenant(tenant), exact=True).count


def _tenant(tenant: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant))])

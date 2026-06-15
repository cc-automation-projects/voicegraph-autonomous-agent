from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.voicegraph.schemas import MemoryFact

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "user_memory"
VECTOR_SIZE = 768


class QdrantMemoryStore:
    def __init__(self, host: str = "qdrant-service", port: int = 6333):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.collection_name = MEMORY_COLLECTION

    async def ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name not in existing:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Коллекция {self.collection_name} создана")
        else:
            logger.info(f"Коллекция {self.collection_name} уже существует")

    async def upsert_memory(self, memory: MemoryFact) -> None:
        point_id = str(uuid4())
        payload: Dict[str, Any] = {
            "user_id": memory.user_id,
            "fact": memory.fact,
            "category": memory.category,
            "confidence": memory.confidence,
            "created_at": memory.created_at.isoformat(),
            "source": memory.source,
        }

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=point_id, vector=memory.embedding or [0.0] * VECTOR_SIZE, payload=payload)],
        )

    async def query_memory(self, user_id: str, query_vector: List[float], top_k: int = 10) -> List[MemoryFact]:
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=top_k,
            with_payload=True,
        )

        facts: List[MemoryFact] = []
        for scored_point in results:
            payload = scored_point.payload or {}
            facts.append(
                MemoryFact(
                    user_id=payload.get("user_id", user_id),
                    fact=payload.get("fact", ""),
                    category=payload.get("category", "general"),
                    confidence=payload.get("confidence", 0.5),
                    embedding=scored_point.vector,
                    source=payload.get("source", "unknown"),
                )
            )
        return facts

    async def delete_user_memory(self, user_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                )
            ),
        )

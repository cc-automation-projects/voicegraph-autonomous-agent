import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional

from mem0 import Memory

from src.voicegraph.schemas import MemoryFact

logger = logging.getLogger(__name__)

_mem0_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem0_worker")

class MemoryManager:
    def __init__(self, decay_lambda: float = 0.1):
        self.decay_lambda = decay_lambda
        self._memory: Optional[Memory] = None
        self._initialized = False

    async def init(self, qdrant_host: str = "qdrant-service", qdrant_port: int = 6333,
                   llm_base_url: str = "http://vllm-service:8000/v1", llm_model: str = "qwen2.5-72b"):
        loop = asyncio.get_running_loop()
        self._memory = await loop.run_in_executor(
            None,
            lambda: Memory.from_config({
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": qdrant_host,
                        "port": qdrant_port,
                        "collection_name": "user_memory",
                        "embedding_model_dims": 768,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm_model,
                        "base_url": llm_base_url,
                        "api_key": "dummy-key",
                        "temperature": 0.1,
                    },
                },
            })
        )
        self._initialized = True
        logger.info("MemoryManager initialized")

    async def save_memory(self, memory: MemoryFact) -> None:
        if not self._initialized:
            await self.init()
        loop = asyncio.get_running_loop()
        metadata = {
            "category": memory.category,
            "source": memory.source,
            "confidence": memory.confidence,
            "timestamp": memory.created_at.isoformat()
        }
        await loop.run_in_executor(
            _mem0_executor,
            self._memory.add,
            memory.fact,
            memory.user_id,
            metadata
        )
        logger.info(f"Saved memory for {memory.user_id}")

    async def retrieve_relevant(self, user_id: str, query: str, top_k: int = 10) -> List[MemoryFact]:
        if not self._initialized:
            await self.init()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _mem0_executor,
            lambda: self._memory.search(query, user_id=user_id, limit=top_k)
        )
        facts = []
        now = datetime.now(timezone.utc)
        for item in results:
            metadata = item.get("metadata", {})
            created_at_str = metadata.get("timestamp") or metadata.get("created_at")
            try:
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
            except (ValueError, TypeError):
                created_at = datetime.now(timezone.utc)
            days_since = (now - created_at).days
            decay = math.exp(-self.decay_lambda * days_since)
            confidence = metadata.get("confidence", 0.5) * decay
            facts.append(
                MemoryFact(
                    user_id=user_id,
                    fact=item.get("memory", item.get("text", "")),
                    category=metadata.get("category", "general"),
                    confidence=confidence,
                    source=metadata.get("source", "mem0"),
                    created_at=created_at,
                )
            )
        facts.sort(key=lambda f: f.confidence, reverse=True)
        return facts

    async def shutdown(self):
        _mem0_executor.shutdown(wait=True)
        logger.info("MemoryManager shut down")

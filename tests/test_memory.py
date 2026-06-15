from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.manager import MemoryManager
from src.memory.qdrant_setup import VECTOR_SIZE
from src.voicegraph.schemas import MemoryFact


class TestMemorySchema:
    def test_memory_fact_defaults(self):
        fact = MemoryFact(
            user_id="00000000-0000-0000-0000-000000000001",
            fact="Test fact",
            category="PREFERENCE",
            confidence=0.9,
            source="test",
        )
        assert fact.embedding is None
        assert fact.created_at is not None

    def test_memory_fact_with_embedding(self):
        fact = MemoryFact(
            user_id="00000000-0000-0000-0000-000000000001",
            fact="Test",
            category="COMPLAINT",
            confidence=0.9,
            embedding=[0.1] * VECTOR_SIZE,
            source="test",
        )
        assert len(fact.embedding) == VECTOR_SIZE


class TestMemoryManager:
    @pytest.mark.asyncio
    @patch("src.memory.manager.Memory.from_config")
    async def test_save_memory(self, mock_mem0_config):
        mock_mem0_instance = MagicMock()
        mock_mem0_instance.add = MagicMock()
        mock_mem0_config.return_value = mock_mem0_instance

        manager = MemoryManager()
        fact = MemoryFact(
            user_id="user-123",
            fact="Не звонить утром",
            category="PREFERENCE",
            confidence=0.95,
            source="voice_worker",
        )

        await manager.save_memory(fact)

        mock_mem0_instance.add.assert_called_once()
        call_args = mock_mem0_instance.add.call_args
        assert call_args[0][0] == "Не звонить утром"
        assert call_args[0][1] == "user-123"

    @pytest.mark.asyncio
    @patch("src.memory.manager.Memory.from_config")
    async def test_retrieve_relevant(self, mock_mem0_config):
        mock_mem0_instance = MagicMock()
        mock_mem0_instance.search.return_value = [
            {
                "memory": "Жаловался на доставку",
                "metadata": {
                    "category": "COMPLAINT",
                    "confidence": 0.8,
                    "source": "reflection_agent",
                    "created_at": "2026-06-01T10:00:00+00:00"
                }
            }
        ]
        mock_mem0_config.return_value = mock_mem0_instance

        manager = MemoryManager()
        facts = await manager.retrieve_relevant(user_id="user-123", query="доставка", top_k=5)

        assert len(facts) == 1
        assert facts[0].category == "COMPLAINT"
        assert facts[0].fact == "Жаловался на доставку"

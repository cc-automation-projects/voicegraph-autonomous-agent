from __future__ import annotations

import logging

from src.memory.manager import MemoryManager
from src.voicegraph.schemas import MemoryFact

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, memory_manager: MemoryManager, max_facts: int = 15):
        self.memory_manager = memory_manager
        self.max_facts = max_facts

    async def build_context(self, user_id: str, conversation_history: str) -> str:
        facts = await self.memory_manager.retrieve_relevant(
            user_id=user_id,
            query=conversation_history,
            top_k=self.max_facts,
        )

        if not facts:
            return ""

        context_lines = []
        for fact in facts:
            context_lines.append(f"- [{fact.category}] {fact.fact} (уверенность: {fact.confidence:.2f})")

        header = f"Контекст пользователя {user_id} (извлечено {len(facts)} фактов):\n"
        return header + "\n".join(context_lines)

    async def update_from_conversation(self, user_id: str, user_text: str, agent_text: str) -> None:
        from src.pii_sanitizer.service import sanitizer

        safe_user_text = sanitizer.sanitize(user_text)

        opinion_fact = MemoryFact(
            user_id=user_id,
            fact=f"Пользователь сказал: {safe_user_text[:200]}",
            category="conversation",
            confidence=0.7,
            source="voice_worker",
        )
        await self.memory_manager.save_memory(opinion_fact)

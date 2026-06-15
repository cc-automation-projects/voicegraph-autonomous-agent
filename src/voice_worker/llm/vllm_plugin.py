from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

import httpx
from livekit.agents import llm, utils

from src.pii_sanitizer.service import sanitizer

logger = logging.getLogger(__name__)

class VLLMPlugin(llm.LLM):
    def __init__(self, *, api_url: str, model: str = "qwen2.5-72b", max_retries: int = 2):
        self._api_url = api_url
        self._model = model
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
        )

    async def chat(self, *, chat_ctx: llm.ChatContext,
                   fnc_ctx: Optional[llm.FunctionContext] = None) -> "VLLMChatStream":
        messages = []
        for msg in chat_ctx.messages:
            role = "assistant" if msg.role == "assistant" else "user"
            safe_content = sanitizer.sanitize(msg.content) if isinstance(msg.content, str) else msg.content
            messages.append({"role": role, "content": safe_content})
        return VLLMChatStream(self._client, self._api_url, self._model, messages, self._max_retries)

    async def aclose(self):
        await self._client.aclose()


class VLLMChatStream(llm.ChatStream):
    def __init__(self, client: httpx.AsyncClient, api_url: str, model: str, messages: List[dict], max_retries: int):
        super().__init__()
        self._client = client
        self._api_url = api_url
        self._model = model
        self._messages = messages
        self._max_retries = max_retries
        self._queue = utils.aio.Chan[llm.ChatChunk]()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 150,
        }
        for attempt in range(self._max_retries):
            try:
                async with self._client.stream(
                    "POST", f"{self._api_url}/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data = json.loads(line[6:])
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    self._queue.send_nowait(llm.ChatChunk(
                                        choices=[llm.Choice(delta=llm.ChoiceDelta(content=delta, role="assistant"))],
                                    ))
                    break
            except Exception as e:
                logger.warning(f"LLM stream attempt {attempt+1} failed: {e}")
                if attempt == self._max_retries - 1:
                    self._queue.send_nowait(llm.ChatChunk(
                        choices=[llm.Choice(delta=llm.ChoiceDelta(
                            content="Извините, сейчас технические неполадки. Попробуйте позже.",
                            role="assistant"
                        ))],
                    ))
                else:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        self._queue.close()

    async def aclose(self) -> None:
        self._queue.close()
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

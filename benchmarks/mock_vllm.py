from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock vLLM Server")

MOCK_LATENCY_MS = int(os.getenv("MOCK_LATENCY_MS", "300"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    await asyncio.sleep(MOCK_LATENCY_MS / 1000.0)

    response_text = "Здравствуйте! Спасибо, что уделили время. Я хотел бы обсудить наше новое предложение."

    if stream:

        async def generate() -> AsyncGenerator[str, None]:
            for char in response_text:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': char, 'role': 'assistant'}, 'index': 0}]})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}, "index": 0}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }


@app.post("/v1/embeddings")
async def embeddings():
    await asyncio.sleep(0.05)
    return {
        "data": [{"embedding": [0.01] * 768, "index": 0}],
        "model": "qwen2.5-72b-mock",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "mock-vllm"}


@app.get("/metrics")
async def metrics():
    return {"status": "ok"}

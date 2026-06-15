from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.voicegraph.observability.metrics import VOICE_LATENCY_MS

logger = logging.getLogger(__name__)

app = FastAPI(title="YandexGPT Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YANDEXGPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEXGPT_CATALOG_ID = ""
YANDEXGPT_API_KEY = ""


class ChatMessage(BaseModel):
    role: str
    text: str


class CompletionRequest(BaseModel):
    model: str = "yandexgpt-lite"
    messages: List[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 2000
    stream: bool = False


class CompletionResponse(BaseModel):
    alternatives: List[Dict[str, Any]]
    usage: Dict[str, int]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "yandexgpt-gateway"}


@app.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest):
    if not YANDEXGPT_API_KEY:
        raise HTTPException(status_code=503, detail="YandexGPT не настроен (API key)")

    start_time = time.monotonic()

    prompt = {
        "modelUri": f"gpt://{YANDEXGPT_CATALOG_ID}/{request.model}",
        "completionOptions": {
            "stream": request.stream,
            "temperature": request.temperature,
            "maxTokens": request.max_tokens,
        },
        "messages": [msg.model_dump() for msg in request.messages],
    }

    if request.stream:

        async def generate() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream(
                    "POST",
                    YANDEXGPT_API_URL,
                    headers={
                        "Authorization": f"Api-Key {YANDEXGPT_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=prompt,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                YANDEXGPT_API_URL,
                headers={
                    "Authorization": f"Api-Key {YANDEXGPT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=prompt,
            )
            response.raise_for_status()
            result = response.json()

            latency_ms = (time.monotonic() - start_time) * 1000
            VOICE_LATENCY_MS.labels(model=request.model, provider="yandexgpt").observe(latency_ms)

            return {"choices": result.get("alternatives", []), "usage": result.get("usage", {})}

    except httpx.HTTPStatusError as e:
        logger.error(f"YandexGPT API error: {e.response.status_code} {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="YandexGPT API error")
    except Exception as e:
        logger.error(f"YandexGPT connection error: {e}")
        raise HTTPException(status_code=502, detail="YandexGPT connection error")

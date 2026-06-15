import asyncio
import json
import logging
from typing import Any, Dict, List

import httpx

from src.pii_sanitizer.service import sanitizer
from src.reflection.schemas import ReflectionOutput

logger = logging.getLogger(__name__)


def recursive_mask(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recursive_mask(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_mask(i) for i in obj]
    if isinstance(obj, str):
        return sanitizer.sanitize(obj)
    return obj


class LLMAnalyzer:
    def __init__(self, api_url: str = "http://vllm-service:8000/v1/chat/completions", max_retries: int = 3):
        self.api_url = api_url
        self.max_retries = max_retries

    async def analyze(self, events: List[Dict[str, Any]]) -> ReflectionOutput:
        safe_events = recursive_mask(events)
        prompt = (
            "Ты — аналитик кампаний VoiceGraph. Проанализируй следующие события и предоставь: "
            "summary (краткое резюме), action_items (список действий), sentiment_trend (positive/neutral/negative), "
            "priority (high/medium/low).\n\n"
            f"События: {json.dumps(safe_events, ensure_ascii=False, default=str)}\n\n"
            "Ответь строго в формате JSON: "
            '{"summary": "...", "action_items": [...], '
            '"sentiment_trend": "...", "priority": "..."}'
        )
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.api_url,
                        json={
                            "model": "qwen2.5-72b",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 1000,
                            "response_format": {"type": "json_object"},
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return ReflectionOutput(
                        insight_id="",
                        campaign_id=safe_events[0].get("campaign_id", "unknown"),
                        summary=parsed.get("summary", "No analysis available"),
                        action_items=parsed.get("action_items", []),
                        sentiment_trend=parsed.get("sentiment_trend", "neutral"),
                        priority=parsed.get("priority", "medium"),
                    )
            except Exception as e:
                logger.warning(f"LLM analyze attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return ReflectionOutput(
                        insight_id="",
                        campaign_id=safe_events[0].get("campaign_id", "unknown"),
                        summary=f"Analysis failed after {self.max_retries} attempts",
                        action_items=["Check LLM service"],
                        sentiment_trend="neutral",
                        priority="low",
                    )
                await asyncio.sleep(0.5 * (2 ** attempt))

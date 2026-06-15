from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ReflectionInput(BaseModel):
    campaign_id: str
    user_id: Optional[str] = None
    event_type: str
    event_payload: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReflectionOutput(BaseModel):
    insight_id: str
    campaign_id: str
    summary: str
    action_items: list[str] = Field(default_factory=list)
    sentiment_trend: str = "neutral"
    priority: str = "medium"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

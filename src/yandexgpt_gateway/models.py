from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    text: str


class CompletionOptions(BaseModel):
    stream: bool = False
    temperature: float = 0.6
    max_tokens: int = 2000


class YandexGPTRequest(BaseModel):
    model_uri: str
    completion_options: CompletionOptions = Field(default_factory=CompletionOptions)
    messages: List[Message]


class Alternative(BaseModel):
    message: Message
    status: str = "ALTERNATIVE_STATUS_FINAL"


class Usage(BaseModel):
    input_text_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class YandexGPTResponse(BaseModel):
    alternatives: List[Alternative]
    usage: Optional[Usage] = None

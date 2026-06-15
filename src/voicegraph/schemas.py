from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class CampaignStateSchema(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    campaign_id: str = ""
    campaign_name: str = ""
    target_goal: str = ""
    candidate_pool: List[Dict[str, Any]] = Field(default_factory=list)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    current_batch: List[Dict[str, Any]] = Field(default_factory=list)
    active_scripts: List[Dict[str, str]] = Field(default_factory=list)
    bandit_weights: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    bandit_state: Optional[Dict[str, Any]] = None
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    phase: str = "scheduling"
    batch_size: int = 50
    current_user_index: int = Field(default=0)
    completed_calls: int = 0
    total_calls_planned: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    budget_limit: float = 0.0
    reflection_insights: List[str] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None)


class ScoredUser(BaseModel):
    user_id: str
    p_answer: float = 0.0
    p_conversion: float = 0.0
    priority_score: float = 0.0
    recommended_call_window: str = "18:00-20:00"
    full_name: str = ""
    phone_hash: str = ""
    region: str = ""
    product_interest: str = ""
    days_since_contact: int = 0
    score: float = 0.0
    total_calls: int = 0
    successful_calls: int = 0
    avg_call_duration_sec: float = 0.0
    last_call_outcome: str = "unknown"
    conversion_history_count: int = 0
    days_since_last_conversion: int = 9999
    campaign_response_rate: float = 0.0
    previous_approval_rate: float = 0.0
    age: int = 35
    credit_score: float = 0.5


class PredictRequest(BaseModel):
    campaign_id: str
    users: List[ScoredUser]


class PredictResponse(BaseModel):
    scored_users: List[ScoredUser]
    threshold: float = 0.3
    total_candidates: int = 0


class UpdateCRMRecordInput(BaseModel):
    user_id: str
    action: Literal["CREATE_DEAL", "UPDATE_FIELD", "CREATE_TASK"]
    nps_score: Optional[int] = Field(default=None, ge=0, le=10)
    notes_masked: str
    idempotency_key: str

    @field_validator("notes_masked")
    @classmethod
    def check_pii(cls, v: str) -> str:
        if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", v):
            raise ValueError("Обнаружен потенциальный номер карты в notes_masked. Используйте PII Sanitizer.")
        return v


class ReflectionInsight(BaseModel):
    session_id: str
    root_cause: Literal["PRICING", "WRONG_TIMING", "AGENT_TONE", "TECHNICAL_ISSUE", "UNKNOWN"]
    suggested_script_tweak: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    direct_quote_from_client: Optional[str] = Field(default=None)


class MemoryFact(BaseModel):
    user_id: str
    fact: str
    category: Literal["PREFERENCE", "COMPLAINT", "DEMOGRAPHIC", "TECHNICAL", "conversation", "general"]
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    source: str = "unknown"
    created_at: datetime = Field(default_factory=datetime.now)


class CampaignDataSchema(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str
    phone_hash: str
    consent_to_call: bool
    last_contact_date: datetime
    ltv_segment: str

    @field_validator("user_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
        if not uuid_pattern.match(v):
            raise ValueError(f"Невалидный формат UUID: {v}")
        return v.lower()

    @field_validator("phone_hash")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not re.match(r"^[a-f0-9]{64}$", v.lower()):
            raise ValueError(f"phone_hash должен быть валидной строкой SHA-256: {v}")
        return v.lower()

    @classmethod
    def hash_phone(cls, phone: str) -> str:
        clean_phone = re.sub(r"\D", "", str(phone))
        return hashlib.sha256(clean_phone.encode("utf-8")).hexdigest()

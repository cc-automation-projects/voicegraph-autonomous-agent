from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.voicegraph.schemas import CampaignStateSchema


class AgentState(BaseModel):
    campaign: CampaignStateSchema = Field(default_factory=CampaignStateSchema)
    reflection_insights: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

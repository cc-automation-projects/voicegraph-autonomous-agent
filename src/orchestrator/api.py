from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.orchestrator.graph_builder import create_campaign_graph
from src.orchestrator.state import AgentState
from src.voicegraph.schemas import CampaignStateSchema

logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceGraph Orchestrator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

campaign_graph = create_campaign_graph()


class LaunchCampaignRequest(BaseModel):
    campaign_id: str
    campaign_name: str
    candidates: list[Dict[str, Any]]
    batch_size: int = 50
    budget_limit: float = 50000.0


class CampaignStatusResponse(BaseModel):
    campaign_id: str
    phase: str
    completed_calls: int
    total_calls_planned: int
    success_count: int
    failure_count: int
    total_revenue: float
    total_cost: float
    roi: float


@app.post("/campaigns/launch")
async def launch_campaign(req: LaunchCampaignRequest):
    initial_state = AgentState(
        campaign=CampaignStateSchema(
            campaign_id=req.campaign_id,
            campaign_name=req.campaign_name,
            candidates=req.candidates,
            batch_size=req.batch_size,
            budget_limit=req.budget_limit,
        )
    )
    config = {"configurable": {"thread_id": req.campaign_id}}
    try:
        await campaign_graph.ainvoke(initial_state, config=config)
        return {"status": "campaign_launched", "campaign_id": req.campaign_id}
    except Exception as e:
        logger.exception("Error executing campaign graph")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/campaigns/{campaign_id}/status", response_model=CampaignStatusResponse)
async def campaign_status(campaign_id: str):
    config = {"configurable": {"thread_id": campaign_id}}
    state = await campaign_graph.aget_state(config)
    if state is None or state.values is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    s = AgentState(**state.values).campaign
    return CampaignStatusResponse(
        campaign_id=s.campaign_id,
        phase=s.phase,
        completed_calls=s.completed_calls,
        total_calls_planned=s.total_calls_planned,
        success_count=s.success_count,
        failure_count=s.failure_count,
        total_revenue=s.total_revenue,
        total_cost=s.total_cost,
        roi=s.total_revenue / max(s.total_cost, 0.01),
    )

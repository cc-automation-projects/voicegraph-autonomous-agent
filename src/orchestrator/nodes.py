from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import asyncpg
import httpx
import redis.asyncio as redis

from src.orchestrator.state import AgentState
from src.voicegraph.config import settings
from src.voicegraph.observability.audit import audit_log

logger = logging.getLogger(__name__)

PROPENSITY_API_URL = "http://propensity-inference:8000"
CALL_STREAM = "campaign:calls"
CONSUMER_GROUP = "voice-workers"

_db_pool: Optional[asyncpg.Pool] = None
_db_pool_lock = asyncio.Lock()

async def get_db_pool() -> asyncpg.Pool:
    global _db_pool
    async with _db_pool_lock:
        if _db_pool is None:
            _db_pool = await asyncpg.create_pool(settings.database_url)
    return _db_pool

async def _batch_frequency_check(user_ids: List[str], db_pool: asyncpg.Pool) -> Dict[str, bool]:
    if not user_ids:
        return {}
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    query = """
        SELECT user_id, COUNT(*) as cnt
        FROM call_logs
        WHERE user_id = ANY($1) AND created_at >= $2
        GROUP BY user_id
    """
    rows = await db_pool.fetch(query, user_ids, seven_days_ago)
    count_map = {row["user_id"]: row["cnt"] for row in rows}
    return {uid: count_map.get(uid, 0) < 2 for uid in user_ids}

async def score_candidates_node(state: AgentState) -> Dict[str, Any]:
    valid_candidates = [c for c in state.campaign.candidates if c.get("consent_to_call") is True]
    if not valid_candidates:
        return {"campaign": {"phase": "completed_no_consent"}}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PROPENSITY_API_URL}/predict",
                json={"campaign_id": state.campaign.campaign_id, "users": valid_candidates}
            )
            response.raise_for_status()
            result = response.json()
            scored_candidates = result.get("scored_users", valid_candidates)
    except Exception as e:
        logger.error(f"Propensity API error: {e}, using fallback order")
        scored_candidates = sorted(valid_candidates, key=lambda u: u.get("score", 0.5), reverse=True)

    batch_size = state.campaign.batch_size
    current_batch = scored_candidates[:batch_size]
    remaining_candidates = scored_candidates[batch_size:]

    return {
        "campaign": {
            "candidates": scored_candidates,
            "current_batch": current_batch,
            "candidate_pool": remaining_candidates,
            "phase": "scoring_complete"
        }
    }


async def schedule_calls_node(state: AgentState) -> Dict[str, Any]:
    batch = state.campaign.current_batch
    if not batch:
        return {"campaign": {"phase": "scheduled"}}

    pool = await get_db_pool()
    user_ids = [u.get("user_id") for u in batch]
    allowed_map = await _batch_frequency_check(user_ids, pool)

    tasks = []
    valid_batch = []
    for user in batch:
        user_id = user.get("user_id")
        if not allowed_map.get(user_id, False):
            logger.info(f"Skipping {user_id} due to frequency cap")
            continue
        valid_batch.append(user)
        task = {
            "user_id": user_id,
            "phone_hash": user.get("phone_hash", ""),
            "script_id": user.get("script_id", "default"),
            "campaign_id": state.campaign.campaign_id,
            "priority_score": user.get("score", 0.5)
        }
        tasks.append(task)

    if tasks:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        try:
            await r.xgroup_create(CALL_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        except redis.ResponseError:
            pass
        async with r.pipeline() as pipe:
            for task in tasks:
                pipe.xadd(CALL_STREAM, task, maxlen=100000)
            await pipe.execute()
        await r.close()

    return {
        "campaign": {
            "current_batch": valid_batch,
            "total_calls_planned": len(valid_batch),
            "phase": "scheduled"
        }
    }


@audit_log(step_name="execute_calls_node")
async def execute_calls_node(state: AgentState) -> Dict[str, Any]:
    if not state.campaign.current_batch:
        return {"campaign": {"phase": "execution_complete"}}
    return {"campaign": {"phase": "execution_in_progress"}}


async def reflect_node(state: AgentState) -> Dict[str, Any]:
    outcome_summary = (
        f"Campaign {state.campaign.campaign_name}: "
        f"{state.campaign.completed_calls} calls, "
        f"{state.campaign.success_count} successes, "
        f"{state.campaign.failure_count} failures."
    )
    logger.info(f"Reflection: {outcome_summary}")
    return {"campaign": {"phase": "reflection_done"}}


def should_continue_campaign(state: AgentState) -> Literal["continue", "end"]:
    if state.campaign.current_batch or state.campaign.candidates:
        return "continue"
    return "end"


def should_continue_batch(state: AgentState) -> Literal["continue", "end"]:
    if state.campaign.candidate_pool:
        return "continue"
    return "end"

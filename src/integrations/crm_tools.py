from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from src.integrations.circuit_breaker import CircuitBreaker
from src.voicegraph.observability.audit import audit_log
from src.voicegraph.schemas import UpdateCRMRecordInput

logger = logging.getLogger(__name__)


class CRMConnector:
    def __init__(self, api_base_url: str = "https://your-company.amocrm.ru", api_token: str = ""):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_token = api_token
        self.circuit_breaker = CircuitBreaker(
            name="amoCRM", failure_threshold=3, recovery_timeout_sec=60
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

    async def close(self):
        await self._client.aclose()

    @audit_log(step_name="crm_update_record")
    async def update_record(self, input_data: UpdateCRMRecordInput) -> Dict[str, Any]:
        async def _do_update() -> Dict[str, Any]:
            for attempt in range(3):
                response = await self._client.post(
                    f"{self.api_base_url}/api/v4/leads/{input_data.user_id}/notes",
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                        "Idempotency-Key": input_data.idempotency_key,
                    },
                    json={
                        "note_type": "call_outcome",
                        "params": {
                            "text": input_data.notes_masked,
                            "nps_score": input_data.nps_score,
                        },
                    },
                )
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited by CRM, retry in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json()
            raise Exception("CRM rate limit exceeded after retries")

        try:
            return await self.circuit_breaker.call(_do_update)
        except Exception as e:
            logger.error(f"Failed to update CRM for user {input_data.user_id}: {e}")
            return {"status": "error", "detail": str(e)}

    @audit_log(step_name="crm_fetch_lead")
    async def fetch_lead(self, lead_id: str) -> Optional[Dict[str, Any]]:
        async def _do_fetch() -> Dict[str, Any]:
            response = await self._client.get(
                f"{self.api_base_url}/api/v4/leads/{lead_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
            )
            response.raise_for_status()
            return response.json()

        try:
            return await self.circuit_breaker.call(_do_fetch)
        except Exception as e:
            logger.error(f"Failed to fetch lead {lead_id}: {e}")
            return None

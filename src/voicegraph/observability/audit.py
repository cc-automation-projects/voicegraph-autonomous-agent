import json
import logging
from functools import wraps
from typing import Any, Callable, Optional
from uuid import uuid4

import asyncpg

from src.pii_sanitizer.service import sanitizer

logger = logging.getLogger(__name__)

_audit_db_pool: Optional[asyncpg.Pool] = None

def set_audit_db_pool(pool: asyncpg.Pool):
    global _audit_db_pool
    _audit_db_pool = pool

def audit_log(step_name: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id", kwargs.get("user_id", "unknown"))
            input_payload = _mask_payload(kwargs)

            try:
                result = await func(*args, **kwargs)
                await _write_audit(
                    session_id=session_id,
                    step_name=step_name,
                    input_payload=input_payload,
                    output_payload=_mask_payload(result),
                    is_success=True,
                )
                return result
            except Exception as e:
                await _write_audit(
                    session_id=session_id,
                    step_name=step_name,
                    input_payload=input_payload,
                    output_payload={"error": str(e)},
                    is_success=False,
                )
                raise
        return wrapper
    return decorator

def _mask_payload(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _mask_payload(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask_payload(item) for item in data]
    if isinstance(data, str):
        return sanitizer.sanitize(data)
    return str(data)

async def _write_audit(session_id: str, step_name: str,
                        input_payload: Any, output_payload: Any,
                        is_success: bool) -> None:
    if _audit_db_pool is None:
        logger.warning("Audit DB pool not initialized. Skipping audit log.")
        return

    try:
        await _audit_db_pool.execute(
            """
            INSERT INTO agent_audit_logs
                (id, session_id, step_name, input_payload, output_payload, is_success, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, NOW())
            """,
            str(uuid4()),
            session_id,
            step_name,
            json.dumps(input_payload, ensure_ascii=False),
            json.dumps(output_payload, ensure_ascii=False),
            is_success,
        )
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

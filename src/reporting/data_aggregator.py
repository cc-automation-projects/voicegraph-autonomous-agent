from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import asyncpg

logger = logging.getLogger(__name__)


class DataAggregator:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool

    async def campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        row = await self.db.fetchrow(
            """
            SELECT
                COUNT(*) as total_calls,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN outcome = 'agreement' THEN 1 ELSE 0 END) as agreements,
                AVG(call_duration) as avg_duration,
                AVG(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as conversion_rate
            FROM call_logs
            WHERE campaign_id = $1
            """,
            campaign_id,
        )
        if row is None:
            return {}
        return {
            "campaign_id": campaign_id,
            "total_calls": row["total_calls"] or 0,
            "successful_calls": row["successful"] or 0,
            "agreements": row["agreements"] or 0,
            "avg_duration_sec": round(float(row["avg_duration"] or 0.0), 2),
            "conversion_rate": round(float(row["conversion_rate"] or 0.0), 4),
        }

    async def script_conversion_stats(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = await self.db.fetch(
            """
            SELECT
                script_id,
                COUNT(*) as total_calls,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_count,
                AVG(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as conversion_rate
            FROM call_logs
            WHERE campaign_id = $1 AND script_id IS NOT NULL
            GROUP BY script_id
            ORDER BY conversion_rate DESC
            """,
            campaign_id,
        )
        return [
            {
                "script_id": row["script_id"],
                "total_calls": row["total_calls"],
                "success_count": row["success_count"],
                "conversion_rate": round(float(row["conversion_rate"] or 0.0), 4),
            }
            for row in rows
        ]

    async def weekly_kpi_report(self) -> Dict[str, Any]:
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        rows = await self.db.fetch(
            """
            SELECT
                DATE(created_at) as day,
                COUNT(*) as calls,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN outcome = 'agreement' THEN 1 ELSE 0 END) as agreements,
                AVG(call_duration) as avg_dur
            FROM call_logs
            WHERE created_at >= $1
            GROUP BY DATE(created_at)
            ORDER BY day
            """,
            week_ago,
        )
        daily_stats = [
            {
                "date": row["day"].isoformat(),
                "calls": row["calls"],
                "success": row["success"],
                "agreements": row["agreements"],
                "avg_duration_sec": round(float(row["avg_dur"] or 0.0), 2),
            }
            for row in rows
        ]
        totals = await self.db.fetchrow(
            """
            SELECT
                COUNT(*) as total_calls,
                COUNT(DISTINCT campaign_id) as active_campaigns,
                SUM(CASE WHEN outcome IN ('success', 'agreement') THEN 1 ELSE 0 END) as total_conversions
            FROM call_logs
            WHERE created_at >= $1
            """,
            week_ago,
        )
        return {
            "period": "weekly",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "daily_stats": daily_stats,
            "total_calls": totals["total_calls"] or 0,
            "active_campaigns": totals["active_campaigns"] or 0,
            "total_conversions": totals["total_conversions"] or 0,
            "overall_conversion_rate": round(
                (totals["total_conversions"] or 0) / max(totals["total_calls"] or 1, 1), 4
            ),
        }

    async def agent_performance(self) -> List[Dict[str, Any]]:
        rows = await self.db.fetch(
            """
            SELECT
                agent_id,
                COUNT(*) as total_calls,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_count,
                AVG(call_duration) as avg_duration
            FROM call_logs
            GROUP BY agent_id
            ORDER BY success_count DESC
            """
        )
        return [
            {
                "agent_id": row["agent_id"],
                "total_calls": row["total_calls"],
                "success_count": row["success_count"],
                "avg_duration_sec": round(float(row["avg_duration"] or 0.0), 2),
            }
            for row in rows
        ]

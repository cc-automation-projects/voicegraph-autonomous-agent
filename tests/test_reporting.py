from __future__ import annotations

import pytest

from src.reporting.delivery_service import ReportDeliveryService


class TestDeliveryService:
    @pytest.mark.asyncio
    async def test_send_report_no_pdf(self):
        svc = ReportDeliveryService()
        result = await svc.send_report(["test@test.com"], "")
        assert result is False

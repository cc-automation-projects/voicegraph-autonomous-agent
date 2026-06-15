from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.pipeline import run_data_pipeline


@pytest.mark.skip(reason="Pre-existing: GX suite 'test_suite' requires physical store initialization")
def test_run_data_pipeline_passes_with_temp_config(monkeypatch):
    test_data = pd.DataFrame([
        {
            "user_id": "00000000-0000-0000-0000-000000000001",
            "phone_hash": "a" * 64,
            "consent_to_call": True,
            "last_contact_date": "2026-06-01",
            "ltv_segment": "PREMIUM",
        }
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.parquet"
        output_path = Path(tmpdir) / "output.parquet"

        test_data.to_parquet(input_path)

        from src.voicegraph.config import settings
        monkeypatch.setattr(settings, "raw_data_path", input_path)
        monkeypatch.setattr(settings, "processed_data_path", output_path)
        monkeypatch.setattr(settings, "gx_expectation_suite_name", "test_suite")

        run_data_pipeline()

        assert output_path.exists()

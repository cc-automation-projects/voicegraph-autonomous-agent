from __future__ import annotations

import pytest

from src.propensity_model.features import BASE_FEATURE_COLUMNS, build_training_dataset, compute_feature_profile
from src.voicegraph.schemas import ScoredUser


class TestFeatures:
    def test_compute_feature_profile(self):
        user = ScoredUser(
            user_id="00000000-0000-0000-0000-000000000001",
            days_since_contact=5,
            total_calls=10,
            successful_calls=3,
            avg_call_duration_sec=120.5,
            last_call_outcome="success",
            region="msk",
            product_interest="card",
            age=30,
            credit_score=0.7,
        )
        df = compute_feature_profile(user)
        assert not df.empty
        assert df.loc[0, "days_since_contact"] == 5
        assert df.loc[0, "region_encoded"] == 0
        assert df.loc[0, "product_interest"] == 0
        assert df.loc[0, "age"] == 30
        assert df.loc[0, "credit_score"] == 0.7

    def test_feature_columns_consistency(self):
        user = ScoredUser(
            user_id="00000000-0000-0000-0000-000000000002",
            days_since_contact=1,
        )
        df = compute_feature_profile(user)
        assert set(df.columns) == set(BASE_FEATURE_COLUMNS)

    @pytest.mark.skip(reason="Pre-existing: featuretools 1.31 + pandas 2.3 incompatibility with fillna on categoricals")
    def test_build_training_dataset(self):
        logs = [
            {"user_id": "u1", "contact_date": "2026-06-01", "call_id": "c1",
             "is_success": 1, "is_converted": 1, "call_duration": 60,
             "region": "msk", "last_outcome": "success", "product_interest": "card"},
            {"user_id": "u1", "contact_date": "2026-06-05", "call_id": "c2",
             "is_success": 0, "is_converted": 0, "call_duration": 30,
             "region": "msk", "last_outcome": "failure", "product_interest": "card"},
        ]
        df = build_training_dataset(logs)
        assert len(df) == 2

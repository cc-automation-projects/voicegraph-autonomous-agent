from __future__ import annotations

import uuid

from src.voicegraph.schemas import (
    ApprovalStatus,
    CampaignStateSchema,
    MemoryFact,
    PredictRequest,
    PredictResponse,
    ScoredUser,
)


class TestScoredUser:
    def test_valid_scored_user(self):
        user = ScoredUser(
            user_id=str(uuid.uuid4()),
            p_answer=0.8,
            p_conversion=0.4,
            priority_score=0.32,
        )
        assert user.p_answer == 0.8
        assert user.priority_score == 0.32

    def test_default_fields(self):
        user = ScoredUser(user_id=str(uuid.uuid4()))
        assert user.score == 0.0
        assert user.days_since_contact == 0
        assert user.recommended_call_window == "18:00-20:00"


class TestPredictRequest:
    def test_valid_request(self):
        req = PredictRequest(
            campaign_id=str(uuid.uuid4()),
            users=[ScoredUser(user_id=str(uuid.uuid4()))],
        )
        assert len(req.users) == 1

    def test_empty_users_allowed(self):
        req = PredictRequest(campaign_id=str(uuid.uuid4()), users=[])
        assert req.users == []


class TestPredictResponse:
    def test_valid_response(self):
        resp = PredictResponse(
            scored_users=[ScoredUser(user_id=str(uuid.uuid4()), score=0.9)],
            threshold=0.3,
            total_candidates=1,
        )
        assert resp.total_candidates == 1
        assert resp.scored_users[0].score == 0.9


class TestMemoryFact:
    def test_valid_memory_fact(self):
        fact = MemoryFact(
            user_id=str(uuid.uuid4()),
            fact="Пользователь интересуется кредитами",
            category="PREFERENCE",
            confidence=0.95,
            source="voice_worker",
        )
        assert fact.category == "PREFERENCE"
        assert fact.embedding is None

    def test_memory_fact_with_embedding(self):
        fact = MemoryFact(
            user_id=str(uuid.uuid4()),
            fact="Test",
            category="COMPLAINT",
            confidence=0.9,
            embedding=[0.1] * 768,
        )
        assert len(fact.embedding) == 768


class TestCampaignStateSchema:
    def test_valid_campaign_state(self):
        state = CampaignStateSchema(
            campaign_id=str(uuid.uuid4()),
        )
        assert state.approval_status == ApprovalStatus.PENDING

    def test_default_factory(self):
        state = CampaignStateSchema(campaign_id=str(uuid.uuid4()))
        assert state.candidate_pool == []
        assert state.error_message is None

from __future__ import annotations

from src.orchestrator.bandit_optimizer import BanditOptimizer
from src.orchestrator.state import AgentState


class TestCampaignState:
    def test_default_state(self):
        state = AgentState()
        assert state.campaign.completed_calls == 0
        assert state.campaign.phase == "scheduling"
        assert state.errors == []

    def test_campaign_state_update(self):
        state = AgentState()
        state.campaign.completed_calls = 10
        state.campaign.success_count = 7
        state.campaign.total_revenue = 10500.0
        state.campaign.total_cost = 50.0
        roi = state.campaign.total_revenue / max(state.campaign.total_cost, 0.01)
        assert roi == 210.0


class TestBanditOptimizer:
    def test_register_arm(self):
        opt = BanditOptimizer()
        opt.register_arm("script_a")
        a, b = opt.get_params("script_a")
        assert a == 1.0
        assert b == 1.0

    def test_update_arm_success(self):
        opt = BanditOptimizer()
        opt.register_arm("script_a")
        opt.update_arm("script_a", 1.0)
        a, b = opt.get_params("script_a")
        assert a == 2.0
        assert b == 1.0

    def test_update_arm_failure(self):
        opt = BanditOptimizer()
        opt.register_arm("script_a")
        opt.update_arm("script_a", 0.0)
        a, b = opt.get_params("script_a")
        assert a == 1.0
        assert b == 2.0

    def test_get_weights(self):
        opt = BanditOptimizer()
        opt.register_arm("script_a")
        opt.register_arm("script_b")
        opt.update_arm("script_a", 1.0)
        opt.update_arm("script_a", 0.0)
        weights = opt.get_weights()
        assert weights["script_a"] == 0.5
        assert weights["script_b"] == 0.5

    def test_select_arm(self):
        opt = BanditOptimizer()
        opt.register_arm("a")
        opt.register_arm("b")
        chosen = opt.select_arm(["a", "b"])
        assert chosen in ("a", "b")

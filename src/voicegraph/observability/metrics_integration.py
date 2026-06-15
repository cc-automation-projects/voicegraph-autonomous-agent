from src.voicegraph.observability.metrics import (
    ASR_WER,
    BANDIT_WEIGHTS,
    CALL_OUTCOMES,
    PII_MASKING_EVENTS,
    VOICE_LATENCY_MS,
)


def record_call_outcome(campaign_id: str, script_id: str, outcome: str):
    CALL_OUTCOMES.labels(campaign_id=campaign_id, script_id=script_id, outcome=outcome).inc()


def update_bandit_weights_metric(campaign_id: str, script_id: str, alpha: float, beta: float):
    BANDIT_WEIGHTS.labels(campaign_id=campaign_id, script_id=script_id, parameter="alpha").set(alpha)
    BANDIT_WEIGHTS.labels(campaign_id=campaign_id, script_id=script_id, parameter="beta").set(beta)


def observe_voice_latency(latency_ms: float, model: str = "vllm", provider: str = "vllm"):
    VOICE_LATENCY_MS.labels(model=model, provider=provider).observe(latency_ms)


def set_asr_wer(wer: float, model_version: str = "distil-large-v3"):
    ASR_WER.labels(model_version=model_version).set(wer)


def increment_pii_masking(pii_type: str):
    PII_MASKING_EVENTS.labels(pii_type=pii_type).inc()

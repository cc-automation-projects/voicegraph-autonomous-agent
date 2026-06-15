import logging

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

VOICE_LATENCY_MS = Histogram(
    "voicegraph_voice_latency_ms",
    "End-to-end latency of the voice pipeline in milliseconds",
    labelnames=["model", "provider"],
    buckets=[100, 300, 500, 800, 1000, 1500, 2000, 3000],
)

CALL_OUTCOMES = Counter(
    "voicegraph_call_outcomes_total",
    "Total number of call outcomes by campaign and script",
    ["campaign_id", "script_id", "outcome"],
)

BANDIT_WEIGHTS = Gauge(
    "voicegraph_bandit_script_weights",
    "Current Thompson Sampling weights (alpha/beta) for script variants",
    ["campaign_id", "script_id", "parameter"],
)

ASR_WER = Gauge(
    "voicegraph_asr_wer",
    "Word Error Rate of the ASR pipeline (0.0 to 1.0)",
    ["model_version"],
)

PII_MASKING_EVENTS = Counter(
    "voicegraph_pii_masked_total",
    "Number of times PII was detected and masked",
    ["pii_type"],
)


def create_prometheus_exporter(port: int = 8001):
    from opentelemetry.exporter.prometheus import PrometheusMetricsExporter
    return PrometheusMetricsExporter(port=port)


def start_metrics_server(port: int = 8001):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


def update_bandit_metrics(campaign_id: str, weights: dict):
    for script_id, params in weights.items():
        BANDIT_WEIGHTS.labels(campaign_id=campaign_id, script_id=script_id, parameter="alpha").set(params["alpha"])
        BANDIT_WEIGHTS.labels(campaign_id=campaign_id, script_id=script_id, parameter="beta").set(params["beta"])

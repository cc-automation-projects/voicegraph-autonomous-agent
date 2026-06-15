import logging

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "voicegraph", otlp_endpoint: str = "http://otel-collector:4317"):
    provider = TracerProvider(
        resource=trace.Resource.create({"service.name": service_name})
    )
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info(f"OpenTelemetry tracing initialized for {service_name}")


def instrument_fastapi(app: FastAPI):
    FastAPIInstrumentor.instrument_app(app)

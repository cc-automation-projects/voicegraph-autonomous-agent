# Contributing to VoiceGraph

## Quick start

```bash
git clone <repo>
cd voicegraph-autonomous-agent
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
python -m spacy download ru_core_news_sm
pre-commit install
```

## Development workflow

1. Create a branch from `develop`
2. Make changes, run `make lint typecheck test`
3. Push and open a PR to `main`

## Code style

- Ruff (E, F, I, N, W), line length 120
- mypy `--strict`
- `pytest` with `asyncio_mode = auto`

## Project layout

```
src/
  orchestrator/     — LangGraph campaign orchestration
  voice_worker/     — LiveKit voice agent pipeline
  propensity_model/ — ML scoring (CatBoost)
  memory/           — Episodic memory (Mem0 + Qdrant)
  reflection/       — Self-learning agent
  integrations/     — CRM, Telegram, Composio, retry worker
  pii_sanitizer/    — PII masking (Presidio)
  data_pipeline/    — ETL + Great Expectations
  yandexgpt_gateway/— YandexGPT API proxy
  voicegraph/       — Shared: config, schemas, observability
```

## Tests

```bash
make test          # unit tests
make integration-test  # integration tests (mark with @pytest.mark.integration)
```

## CI

GitHub Actions runs on every PR: lint (ruff + mypy) → test (with redis + qdrant services).

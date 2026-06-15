# Артефакты данных и контрактов (Data & API Contracts)

## 1.1. ER-диаграмма и DDL-скрипты (Схема БД PostgreSQL)

**Цель:** Обеспечить строгую, нормализованную и производительную схему хранения данных с учетом требований 152-ФЗ (маскирование) и высоких нагрузок на запись (Call Logs, Audit Logs).

### Ключевые сущности и связи:
*   `campaigns` (1) → (N) `call_logs`
*   `users` (1) → (N) `call_logs`
*   `call_logs` (1) → (N) `agent_audit_logs`
*   `campaigns` (1) → (N) `reflection_insights`

### DDL-скрипт (PostgreSQL 16 + pgvector)

```sql
-- Включение необходимых расширений
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- Для полнотекстового поиска по логам
CREATE EXTENSION IF NOT EXISTS "vector";  -- Для интеграции с Mem0/Qdrant метаданными (опционально)

-- 1. Таблица кампаний
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    goal VARCHAR(100) NOT NULL, -- 'NPS', 'COLLECTION', 'CANVASSING'
    status VARCHAR(50) DEFAULT 'DRAFT', -- 'DRAFT', 'PENDING_APPROVAL', 'ACTIVE', 'COMPLETED', 'FAILED'
    bandit_config JSONB NOT NULL DEFAULT '{"alpha": 1, "beta": 1}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_campaigns_status ON campaigns(status);

-- 2. Таблица пользователей (контактов)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_hash VARCHAR(64) UNIQUE NOT NULL, -- Хэш номера телефона (SHA-256)
    consent_to_call BOOLEAN DEFAULT FALSE, -- Строгое требование 38-ФЗ
    ltv_segment VARCHAR(50), -- 'PREMIUM', 'STANDARD', 'LOW'
    last_contact_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_users_consent ON users(consent_to_call) WHERE consent_to_call = TRUE;

-- 3. Таблица логов звонков (High Write Volume)
CREATE TABLE call_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id UUID NOT NULL UNIQUE, -- Идентификатор сессии LiveKit/LangGraph
    script_id VARCHAR(100), -- ID использованного варианта скрипта (для Bandit)
    duration_sec INT,
    outcome VARCHAR(50), -- 'SUCCESS', 'REFUSAL', 'HANGUP', 'ERROR'
    max_sentiment_score VARCHAR(20), -- 'CALM', 'ANNOYED', 'ANGRY'
    transcript_masked TEXT, -- !! ВАЖНО: Только замаскированный текст (PII-sanitized)
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_call_logs_campaign ON call_logs(campaign_id);
CREATE INDEX idx_call_logs_session ON call_logs(session_id);
CREATE INDEX idx_call_logs_outcome ON call_logs(outcome);

-- 4. Таблица аудит-логов действий агента (Неизменяемый лог)
CREATE TABLE agent_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    step_name VARCHAR(100) NOT NULL, -- 'planner_node', 'tool_execution', 'reflection_node'
    tool_name VARCHAR(100), -- 'composio_bx24_create_deal', 'mem0_search'
    input_payload JSONB, -- !! ВАЖНО: Должно быть предварительно обработано Presidio
    output_payload JSONB, -- !! ВАЖНО: Должно быть предварительно обработано Presidio
    llm_reasoning TEXT, -- Краткое обоснование решения LLM
    is_success BOOLEAN
);
CREATE INDEX idx_audit_logs_session ON agent_audit_logs(session_id);
CREATE INDEX idx_audit_logs_tool ON agent_audit_logs(tool_name);
-- GIN индекс для быстрого поиска по JSONB (если потребуется аудит конкретных параметров)
CREATE INDEX idx_audit_logs_input_payload ON agent_audit_logs USING GIN (input_payload);

-- 5. Таблица инсайтов рефлексии
CREATE TABLE reflection_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    root_cause VARCHAR(100) NOT NULL, -- 'PRICING', 'WRONG_TIMING', 'AGENT_TONE', 'TECHNICAL'
    suggested_script_tweak TEXT NOT NULL,
    confidence_score NUMERIC(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    applied_to_campaign BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 1.2. OpenAPI (Swagger) спецификации для микросервисов

**Цель:** Обеспечить четкий контракт между LangGraph Orchestrator, ML Propensity Service и внешними системами (CRM), позволяя командам разрабатывать компоненты параллельно с использованием моков.

### Спецификация 1: ML Propensity Service (`/api/v1/ml/score`)
*Описывает эндпоинт, который LangGraph вызывает для получения приоритизированного списка контактов.*

```yaml
openapi: 3.0.3
info:
  title: VoiceGraph Propensity ML API
  version: 1.0.0
paths:
  /api/v1/ml/score/batch:
    post:
      summary: Рассчитать приоритетный скор для батча пользователей
      description: Возвращает отсортированный список пользователей на основе p_answer и p_conversion.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                campaign_id:
                  type: string
                  format: uuid
                user_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                  maxItems: 1000
      responses:
        '200':
          description: Успешный расчет скоров
          content:
            application/json:
              schema:
                type: object
                properties:
                  scored_users:
                    type: array
                    items:
                      type: object
                      properties:
                        user_id:
                          type: string
                          format: uuid
                        p_answer:
                          type: number
                          format: float
                          example: 0.75
                        p_conversion:
                          type: number
                          format: float
                          example: 0.40
                        priority_score:
                          type: number
                          format: float
                          description: "p_answer * p_conversion"
                          example: 0.30
                        recommended_call_window:
                          type: string
                          description: "Рекомендуемое время звонка, например '18:00-20:00'"
        '500':
          description: Внутренняя ошибка ML-сервиса
```

### Спецификация 2: Webhook для интеграции с CRM (Composio Callback)
*Описывает эндпоинт, который принимает асинхронные подтверждения действий от CRM.*

```yaml
  /api/v1/webhooks/crm-action-result:
    post:
      summary: Получить результат асинхронного действия в CRM
      description: Вызывается Composio/CRM после создания сделки или задачи.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - idempotency_key
                - status
              properties:
                idempotency_key:
                  type: string
                  description: "Уникальный ключ запроса для предотвращения дубликатов"
                action_type:
                  type: string
                  enum: [CREATE_DEAL, UPDATE_CONTACT, CREATE_TASK]
                status:
                  type: string
                  enum: [SUCCESS, FAILED, RATE_LIMITED]
                external_id:
                  type: string
                  description: "ID сущности в Битрикс24/amoCRM"
                error_message:
                  type: string
      responses:
        '200':
          description: Webhook успешно принят
```

---

## 1.3. Строгие Pydantic-модели (Data Contracts)

**Цель:** Обеспечить строгую типизацию состояния LangGraph (`State`) и входных/выходных данных для Function Calling (Tools). Это предотвращает "галлюцинации" LLM в виде невалидного JSON и обеспечивает стабильность графа.

### Файл: `voicegraph/schemas.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# ==========================================
# 1. Модели состояния LangGraph (State)
# ==========================================

class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class CampaignState(BaseModel):
    """Основное состояние графа кампании. Передается между всеми узлами LangGraph."""
    campaign_id: str = Field(description="UUID кампании")
    target_goal: str = Field(description="Цель кампании, например, 'NPS_Optimization'")
    
    # Данные для обзвона
    candidate_pool: List[Dict[str, Any]] = Field(default_factory=list, description="Отсортированный список пользователей со скорами")
    active_scripts: List[Dict[str, str]] = Field(default_factory=list, description="Список вариантов скриптов: [{'id': 'v1', 'text': '...'}]")
    
    # Оптимизация
    bandit_weights: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Параметры alpha/beta для каждого script_id")
    
    # Управление потоком
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, description="Статус одобрения супервайзером")
    current_user_index: int = Field(default=0, description="Индекс текущего пользователя в candidate_pool")
    
    # Рефлексия
    reflection_insights: List[str] = Field(default_factory=list, description="Накопленные инсайты для обновления промптов")
    error_message: Optional[str] = Field(default=None, description="Текст ошибки, если граф упал")

# ==========================================
# 2. Модели для Function Calling (Tools)
# ==========================================

class UpdateCRMRecordInput(BaseModel):
    """Схема ввода для инструмента обновления CRM через Composio."""
    user_id: str = Field(description="Внутренний UUID пользователя в нашей системе")
    action: Literal["CREATE_DEAL", "UPDATE_FIELD", "CREATE_TASK"] = Field(description="Тип действия в CRM")
    nps_score: Optional[int] = Field(default=None, ge=0, le=10, description="Оценка NPS от 0 до 10, если применимо")
    notes_masked: str = Field(description="Краткое саммари звонка. ВАЖНО: Не должно содержать ПДн (паспорта, карты).")
    idempotency_key: str = Field(description="Уникальный ключ запроса (sha256 от user_id + action + timestamp)")

    @field_validator('notes_masked')
    @classmethod
    def check_pii(cls, v: str) -> str:
        # Примечание: Фактическая проверка Presidio должна быть на уровне middleware, 
        # но базовая regex-проверка здесь полезна как fail-fast.
        import re
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', v): # Простая проверка на номер карты
            raise ValueError("Обнаружен потенциальный номер карты в notes_masked. Используйте PII Sanitizer.")
        return v

# ==========================================
# 3. Модели для Reflection Agent
# ==========================================

class ReflectionInsight(BaseModel):
    """Структурированный вывод LLM при анализе неудачного звонка."""
    session_id: str = Field(description="UUID сессии звонка")
    root_cause: Literal["PRICING", "WRONG_TIMING", "AGENT_TONE", "TECHNICAL_ISSUE", "UNKNOWN"] = Field(
        description="Основная причина отказа или негативной реакции"
    )
    suggested_script_tweak: str = Field(
        description="Конкретное, атомарное предложение по изменению текста скрипта или логики"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, 
        description="Уверенность агента в данном инсайте (0.0 - 1.0)"
    )
    direct_quote_from_client: Optional[str] = Field(
        default=None, 
        description="Прямая цитата клиента (замаскированная), подтверждающая инсайт"
    )

# ==========================================
# 4. Модели для Mem0 (Episodic Memory)
# ==========================================

class MemoryFact(BaseModel):
    """Факт, извлекаемый из диалога для сохранения в долговременную память."""
    user_id: str = Field(description="UUID пользователя")
    fact: str = Field(description="Сформулированный факт, например: 'Клиент предпочитает звонки после 18:00'")
    category: Literal["PREFERENCE", "COMPLAINT", "DEMOGRAPHIC", "TECHNICAL"] = Field(description="Категория факта")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в корректности извлеченного факта")
```


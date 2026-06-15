-- =============================================================================
-- VoiceGraph Database Initialization Script
-- =============================================================================

-- 1. Включение необходимых расширений PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 2. Таблица кампаний (базовая)
CREATE TABLE IF NOT EXISTS campaigns (
    id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(64) DEFAULT 'DRAFT',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Таблица пользователей (базовая)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY,
    phone_hash VARCHAR(64) NOT NULL,
    consent_to_call BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Таблица логов звонков (ВАША СХЕМА)
CREATE TABLE IF NOT EXISTS call_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    phone_hash VARCHAR(64) NOT NULL,
    script_id VARCHAR(128),
    campaign_id VARCHAR(128) NOT NULL,
    outcome VARCHAR(64),
    nps_score INTEGER,
    call_duration DOUBLE PRECISION DEFAULT 0.0,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_logs_user_id ON call_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_call_logs_campaign_id ON call_logs(campaign_id);
CREATE INDEX IF NOT EXISTS idx_call_logs_created_at ON call_logs(created_at);

-- 5. Таблица аудит-логов (НЕОБХОДИМА для работы декоратора @audit_log)
CREATE TABLE IF NOT EXISTS agent_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    step_name VARCHAR(128) NOT NULL,
    input_payload JSONB,
    output_payload JSONB,
    is_success BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_logs_session ON agent_audit_logs(session_id);

-- 6. Таблица инсайтов рефлексии (НЕОБХОДИМА для работы Reflection Agent)
CREATE TABLE IF NOT EXISTS reflection_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id VARCHAR(128) NOT NULL,
    root_cause VARCHAR(128),
    suggested_script_tweak TEXT,
    confidence_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reflection_campaign ON reflection_insights(campaign_id);
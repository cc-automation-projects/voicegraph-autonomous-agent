from __future__ import annotations

import logging
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.propensity_model.features import compute_features_batch
from src.voicegraph.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Propensity Model Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: CatBoostClassifier | None = None

MODEL_PATH = Path("models/propensity_model.pkl")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Модель загружена из {MODEL_PATH}")
    else:
        logger.warning("Файл модели не найден. Работа без модели (возврат baseline).")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features_df = compute_features_batch(request.users)
        proba = model.predict_proba(features_df)[:, 1]
        scored_users = []
        for user, p in zip(request.users, proba):
            scored = user.model_copy(update={"score": float(p), "priority_score": float(p), "p_conversion": float(p)})
            scored_users.append(scored)
        scored_users.sort(key=lambda u: u.score, reverse=True)
        return PredictResponse(scored_users=scored_users, threshold=0.3, total_candidates=len(scored_users))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(e))

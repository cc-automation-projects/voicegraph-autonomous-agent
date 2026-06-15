import logging
from typing import List

import featuretools as ft
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.voicegraph.schemas import ScoredUser

logger = logging.getLogger(__name__)

BASE_FEATURE_COLUMNS = [
    "days_since_contact", "total_calls", "successful_calls", "avg_call_duration",
    "last_outcome", "region_encoded", "product_interest", "conversion_history_count",
    "days_since_last_conversion", "campaign_response_rate", "previous_approval_rate",
    "is_weekend", "hour_of_day", "age", "credit_score"
]

def compute_feature_profile(user: ScoredUser) -> pd.DataFrame:
    return compute_features_batch([user])

def compute_features_batch(users: List[ScoredUser]) -> pd.DataFrame:
    rows = []
    for user in users:
        region_map = {"msk": 0, "spb": 1, "region": 2}
        outcome_map = {"success": 0, "failure": 1, "no_answer": 2, "unknown": 3}
        product_map = {"card": 0, "credit": 1, "mortgage": 2, "investment": 3}
        row = {
            "days_since_contact": user.days_since_contact,
            "total_calls": user.total_calls or 0,
            "successful_calls": user.successful_calls or 0,
            "avg_call_duration": user.avg_call_duration_sec or 0.0,
            "last_outcome": outcome_map.get(user.last_call_outcome or "unknown", 3),
            "region_encoded": region_map.get((user.region or "region").lower(), 2),
            "product_interest": product_map.get((user.product_interest or "card").lower(), 0),
            "conversion_history_count": user.conversion_history_count or 0,
            "days_since_last_conversion": user.days_since_last_conversion or 9999,
            "campaign_response_rate": user.campaign_response_rate or 0.0,
            "previous_approval_rate": user.previous_approval_rate or 0.0,
            "is_weekend": 0,
            "hour_of_day": 12,
            "age": user.age or 35,
            "credit_score": user.credit_score or 0.5,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def build_training_dataset(logs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(logs)

    df["days_since_contact"] = (pd.Timestamp.now() - pd.to_datetime(df["contact_date"])).dt.days
    df["is_weekend"] = pd.to_datetime(df["contact_date"]).dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df["hour_of_day"] = pd.to_datetime(df["contact_date"]).dt.hour
    df["avg_call_duration"] = df["call_duration"].fillna(0.0)
    df["total_calls"] = df.groupby("user_id")["call_id"].transform("count")
    df["successful_calls"] = df.groupby("user_id")["is_success"].transform("sum")
    df["conversion_history_count"] = df.groupby("user_id")["is_converted"].transform("sum")

    for col in ["region", "last_outcome", "product_interest"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    es = ft.EntitySet(id="training")
    es.add_dataframe(dataframe_name="logs", dataframe=df, index="_row_id", make_index=True)

    feature_matrix, feature_defs = ft.dfs(
        target_dataframe_name="logs",
        entityset=es,
        max_depth=1,
        verbose=False
    )

    return feature_matrix.fillna(0)

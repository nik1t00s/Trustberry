from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataio import (
    EXPORTS_DIR,
    load_records,
    save_csv,
    save_jsonl,
    validate_prediction_records,
    validate_training_records,
)
from .db import insert_reviews
from .ml import (
    get_available_model_options,
    get_model_type,
    get_model_version,
    load_model,
    load_training_dataframe,
    predict_dataset,
    save_trained_model,
    train_model,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def train_from_path(
    dataset_path: Path,
    *,
    mode: str = "single",
    model_type: str = "logistic_regression",
    training_speed: str = "fast",
) -> dict[str, Any]:
    raw_records = load_records(dataset_path)
    validated_records, validation = validate_training_records(
        raw_records,
        source_file=dataset_path.name,
        deduplicate=True,
    )
    df = load_training_dataframe(validated_records)
    pipeline, metrics_payload = train_model(
        df,
        source_file=dataset_path.name,
        mode=mode,
        model_type=model_type,
        training_speed=training_speed,
    )
    save_trained_model(pipeline, metrics_payload)
    metrics = metrics_payload["summary"]

    stored_rows = insert_reviews(
        [
            {
                "product_id": row.get("product_id"),
                "rating": row.get("rating"),
                "color": row.get("color"),
                "review_text": row.get("review_text"),
                "seller_answer": row.get("seller_answer"),
                "fake_label": row.get("fake_label"),
                "probability_fake": None,
                "predicted_label": None,
            }
            for row in df.to_dict(orient="records")
        ]
    )

    warnings: list[str] = []
    if metrics["f1"] < 0.5:
        warnings.append("Качество модели пока остается умеренным: F1-score ниже 0.50.")
    positive_rate = metrics["positive_rate"]
    if positive_rate < 0.15 or positive_rate > 0.85:
        warnings.append("Датасет заметно несбалансирован по классам.")
    if mode == "compare":
        warnings.append("Активной сохранена модель с лучшим F1-score; при близких значениях использован ROC-AUC.")
    if model_type == "xgboost" and not get_available_model_options()["xgboost"]["available"]:
        warnings.append("XGBoost недоступен в текущем окружении: библиотека не установлена.")

    return {
        "metrics_payload": metrics_payload,
        "metrics": {**metrics, "stored_rows": stored_rows},
        "validation": validation,
        "dataset_summary": {
            "records_after_validation": len(validated_records),
            "class_balance": metrics["class_balance"],
            "positive_rate": metrics["positive_rate"],
            "source_file": dataset_path.name,
        },
        "warnings": warnings,
    }


def predict_from_path(dataset_path: Path) -> dict[str, Any]:
    raw_records = load_records(dataset_path)
    validated_records, validation = validate_prediction_records(raw_records, source_file=dataset_path.name)
    model = load_model()
    processed_at = utc_now()
    model_version = get_model_version()
    model_type = get_model_type()
    predictions = predict_dataset(model, validated_records)

    enriched_rows = [
        {
            "nmId": row.get("nmId"),
            "productValuation": row.get("productValuation"),
            "color": row.get("color"),
            "text": row.get("text"),
            "answer": row.get("answer"),
            "_reviewmark_id": row.get("_reviewmark_id"),
            "_reviewmark_source_index": row.get("_reviewmark_source_index"),
            "predicted_label": row.get("predicted_label"),
            "predicted_proba": row.get("predicted_proba"),
            "model_version": model_version,
            "model_type": model_type,
            "processed_at": processed_at,
            "source_file": dataset_path.name,
        }
        for row in predictions
    ]

    insert_reviews(
        [
            {
                "product_id": row.get("nmId"),
                "rating": row.get("productValuation"),
                "color": row.get("color"),
                "review_text": row.get("text"),
                "seller_answer": row.get("answer"),
                "fake_label": None,
                "probability_fake": row.get("predicted_proba"),
                "predicted_label": row.get("predicted_label"),
            }
            for row in enriched_rows
        ]
    )

    stem = dataset_path.name.replace(".", "_")
    slug = timestamp_slug()
    jsonl_path = EXPORTS_DIR / f"{stem}_{slug}_predictions.jsonl"
    csv_path = EXPORTS_DIR / f"{stem}_{slug}_predictions.csv"
    save_jsonl(jsonl_path, enriched_rows)
    save_csv(csv_path, enriched_rows)

    return {
        "rows": enriched_rows,
        "validation": validation,
        "exports": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
        "processed_at": processed_at,
    }

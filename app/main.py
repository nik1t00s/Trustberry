from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import count_reviews, fetch_recent, init_db, insert_reviews
from .dataio import load_records
from .labeling import apply_decision, create_session, get_next_record, get_progress, list_sessions, load_session, load_source_records
from .ml import METRICS_PATH, get_available_model_options, get_model_type, get_model_version, load_model, predict_one
from .schemas import PredictionOut, ReviewIn, TrainResponse
from .services import predict_from_path, train_from_path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".jl", ".txt"}

app = FastAPI(title="Система обнаружения мошеннических отзывов", version="1.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.on_event("startup")
def startup_event() -> None:
    init_db()


def load_saved_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_index(
    request: Request,
    *,
    result: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    message: str | None = None,
    message_type: str = "info",
    dataset_name: str | None = None,
) -> HTMLResponse:
    saved_metrics = load_saved_metrics()
    saved_summary = (saved_metrics or {}).get("summary", saved_metrics)
    active_model = (saved_metrics or {}).get("active_model", {})
    session_loaded = dataset_name is not None
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "recent": fetch_recent(20),
            "result": result,
            "metrics": metrics if metrics is not None else saved_summary,
            "message": message,
            "message_type": message_type,
            "dataset_name": dataset_name,
            "model_ready": saved_metrics is not None,
            "session_loaded": session_loaded,
            "reviews_total": count_reviews(),
            "active_model": active_model,
        },
    )


def build_risk_badge(probability: float) -> dict[str, str]:
    if probability >= 0.75:
        return {"level": "Высокий риск", "tone": "danger"}
    if probability >= 0.4:
        return {"level": "Средний риск", "tone": "warning"}
    return {"level": "Низкий риск", "tone": "safe"}


def render_training_page(
    request: Request,
    *,
    metrics: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    comparison: dict[str, Any] | None = None,
    message: str | None = None,
    message_type: str = "info",
    dataset_name: str | None = None,
) -> HTMLResponse:
    saved_metrics = load_saved_metrics()
    saved_summary = (saved_metrics or {}).get("summary", saved_metrics)
    return templates.TemplateResponse(
        request=request,
        name="training.html",
        context={
            "request": request,
            "metrics": metrics if metrics is not None else saved_summary,
            "message": message,
            "message_type": message_type,
            "dataset_name": dataset_name,
            "model_ready": saved_metrics is not None,
            "summary": summary or {},
            "warnings": warnings or [],
            "comparison": comparison if comparison is not None else (saved_metrics or {}).get("comparison"),
            "model_options": get_available_model_options(),
        },
    )


def render_prediction_page(
    request: Request,
    *,
    rows: list[dict[str, Any]] | None = None,
    exports: dict[str, str] | None = None,
    summary: dict[str, Any] | None = None,
    message: str | None = None,
    message_type: str = "info",
    dataset_name: str | None = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "request": request,
            "rows": rows or [],
            "exports": exports or {},
            "message": message,
            "message_type": message_type,
            "dataset_name": dataset_name,
            "model_ready": load_saved_metrics() is not None,
            "summary": summary or {},
        },
    )


def render_labeling_page(
    request: Request,
    *,
    session_id: str | None = None,
    message: str | None = None,
    message_type: str = "info",
) -> HTMLResponse:
    sessions = list_sessions()
    session = load_session(session_id) if session_id else (sessions[0] if sessions else None)
    current_record = None
    progress = None

    if session:
        records = load_source_records(session["session_id"])
        current_record = get_next_record(session, records)
        progress = get_progress(session)

    return templates.TemplateResponse(
        request=request,
        name="labeling.html",
        context={
            "request": request,
            "sessions": sessions,
            "session": session,
            "current_record": current_record,
            "progress": progress,
            "message": message,
            "message_type": message_type,
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render_index(request)


@app.get("/labeling", response_class=HTMLResponse)
def labeling_page(request: Request, session_id: str | None = None) -> HTMLResponse:
    return render_labeling_page(request, session_id=session_id)


@app.post("/labeling/start", response_class=HTMLResponse)
def start_labeling_session(request: Request, dataset: UploadFile = File(...)) -> HTMLResponse:
    if not dataset.filename:
        return render_labeling_page(
            request,
            message="Выберите JSONL-файл для ручной разметки.",
            message_type="error",
        )

    dst = UPLOADS_DIR / dataset.filename
    with dst.open("wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        session = create_session(load_records(dst), dataset.filename)
    except Exception as exc:
        return render_labeling_page(
            request,
            message=f"Не удалось открыть файл для разметки: {exc}",
            message_type="error",
        )

    return render_labeling_page(
        request,
        session_id=session["session_id"],
        message="Сессия разметки создана. Автосохранение включено.",
        message_type="success",
    )


@app.post("/labeling/action", response_class=HTMLResponse)
def labeling_action(
    request: Request,
    session_id: str = Form(...),
    record_id: str = Form(...),
    decision: str = Form(...),
) -> HTMLResponse:
    try:
        apply_decision(session_id, record_id, decision)
    except Exception as exc:
        return render_labeling_page(
            request,
            session_id=session_id,
            message=f"Не удалось сохранить разметку: {exc}",
            message_type="error",
        )

    message = (
        "Отзыв помечен как мошеннический."
        if decision == "1"
        else "Отзыв помечен как не мошеннический."
        if decision == "0"
        else "Отзыв пропущен."
    )
    return render_labeling_page(
        request,
        session_id=session_id,
        message=message,
        message_type="success" if decision != "skip" else "info",
    )


@app.get("/training", response_class=HTMLResponse)
def training_page(request: Request) -> HTMLResponse:
    return render_training_page(request)


@app.post("/training/upload", response_class=HTMLResponse)
def training_upload(
    request: Request,
    dataset: UploadFile = File(...),
    training_mode: str = Form(default="single"),
    selected_model: str = Form(default="logistic_regression"),
    training_speed: str = Form(default="fast"),
) -> HTMLResponse:
    if not dataset.filename:
        return render_training_page(
            request,
            message="Выберите файл с размеченными отзывами.",
            message_type="error",
        )

    dst = UPLOADS_DIR / dataset.filename
    with dst.open("wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        training_result = train_from_path(
            dst,
            mode="compare" if training_mode == "compare" else "single",
            model_type=selected_model,
            training_speed=training_speed,
        )
    except Exception as exc:
        return render_training_page(
            request,
            message=f"Не удалось обучить модель: {exc}",
            message_type="error",
            dataset_name=dataset.filename,
        )

    return render_training_page(
        request,
        metrics=training_result["metrics"],
        summary={**training_result["validation"], **training_result["dataset_summary"]},
        warnings=training_result["warnings"],
        comparison=training_result["metrics_payload"].get("comparison"),
        message=(
            f"Обучение завершено на файле {dataset.filename}. "
            f"Использовано {training_result['dataset_summary']['records_after_validation']} размеченных записей."
        ),
        message_type="success",
        dataset_name=dataset.filename,
    )


@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request) -> HTMLResponse:
    return render_prediction_page(request)


@app.post("/prediction/upload", response_class=HTMLResponse)
def prediction_upload(request: Request, dataset: UploadFile = File(...)) -> HTMLResponse:
    if not dataset.filename:
        return render_prediction_page(
            request,
            message="Выберите файл с новыми отзывами для прогнозирования.",
            message_type="error",
        )

    dst = UPLOADS_DIR / dataset.filename
    with dst.open("wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        prediction_result = predict_from_path(dst)
    except Exception as exc:
        return render_prediction_page(
            request,
            message=f"Не удалось выполнить прогноз: {exc}",
            message_type="error",
            dataset_name=dataset.filename,
        )

    predictions = prediction_result["rows"]
    flagged = sum(1 for row in predictions if row["predicted_label"] == 1)
    return render_prediction_page(
        request,
        rows=sorted(predictions[:200], key=lambda row: row["predicted_proba"], reverse=True),
        exports=prediction_result["exports"],
        summary=prediction_result["validation"],
        message=(
            f"Прогноз завершен. Обработано {len(predictions)} записей, "
            f"подозрительных отзывов: {flagged}."
        ),
        message_type="success",
        dataset_name=dataset.filename,
    )


@app.post("/predict", response_model=PredictionOut)
def api_predict(review: ReviewIn) -> PredictionOut:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = predict_one(model, review.model_dump())
    insert_reviews(
        [
            {
                **review.model_dump(),
                "fake_label": None,
                "probability_fake": result["probability_fake"],
                "predicted_label": result["prediction"],
            }
        ]
    )
    return PredictionOut(**result)


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    product_id: int | None = Form(default=None),
    rating: int | None = Form(default=None),
    color: str | None = Form(default=None),
    review_text: str = Form(...),
    seller_answer: str | None = Form(default=None),
) -> HTMLResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        return render_index(request, message=str(exc), message_type="error")

    payload = {
        "product_id": product_id,
        "rating": rating,
        "color": color,
        "review_text": review_text,
        "seller_answer": seller_answer,
    }
    result = predict_one(model, payload)
    result.update(build_risk_badge(result["probability_fake"]))
    insert_reviews(
        [
            {
                **payload,
                "fake_label": None,
                "probability_fake": result["probability_fake"],
                "predicted_label": result["prediction"],
            }
        ]
    )
    return render_index(
        request,
        result=result,
        message=(
            f"Отзыв проанализирован: класс — {result['label']}, "
            f"вероятность мошенничества — {result['probability_fake']:.4f}."
        ),
        message_type="success",
    )


@app.post("/train", response_model=TrainResponse)
def train_from_sample() -> TrainResponse:
    sample_path = DATA_DIR / "labeled_records.jsonl"
    training_result = train_from_path(sample_path, mode="compare", training_speed="fast")
    metrics = training_result["metrics"]
    return TrainResponse(
        message="Модель успешно обучена на демонстрационном наборе",
        train_size=metrics["train_size"],
        test_size=metrics["test_size"],
        metrics={
            **{k: v for k, v in metrics.items() if k not in {"classification_report"}},
        },
    )


@app.post("/train-demo", response_class=HTMLResponse)
def train_demo_form(request: Request) -> HTMLResponse:
    sample_path = DATA_DIR / "labeled_records.jsonl"
    training_result = train_from_path(sample_path, mode="compare", training_speed="fast")
    return render_index(
        request,
        metrics=training_result["metrics"],
        message="Демонстрационный размеченный набор успешно загружен, модель обучена.",
        message_type="success",
        dataset_name=sample_path.name,
    )


@app.post("/upload-and-train", response_class=HTMLResponse)
def upload_and_train(request: Request, dataset: UploadFile = File(...)) -> HTMLResponse:
    if not dataset.filename:
        return render_index(
            request,
            message="Выберите файл датасета перед загрузкой.",
            message_type="error",
        )

    suffixes = [part.lower() for part in Path(dataset.filename).suffixes]
    is_supported = len(suffixes) == 1 and suffixes[0] in SUPPORTED_EXTENSIONS
    if suffixes[-2:] == [".json", ".zst"]:
        is_supported = True

    if not is_supported:
        return render_index(
            request,
            message="Поддерживаются файлы JSON, JSONL, TXT(JSON Lines) и JSON.ZST.",
            message_type="error",
        )

    dst = UPLOADS_DIR / dataset.filename
    with dst.open("wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        training_result = train_from_path(dst, mode="compare", training_speed="fast")
    except Exception as exc:
        return render_index(
            request,
            message=f"Не удалось обработать датасет: {exc}",
            message_type="error",
            dataset_name=dataset.filename,
        )

    return render_index(
        request,
        metrics=training_result["metrics"],
        message=(
            f"Датасет загружен и использован для обучения модели. "
            f"Принято размеченных записей: {training_result['dataset_summary']['records_after_validation']}."
        ),
        message_type="success",
        dataset_name=dataset.filename,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": get_model_version(), "model_type": get_model_type()}


@app.get("/reviews")
def reviews() -> list[dict[str, Any]]:
    return fetch_recent(100)

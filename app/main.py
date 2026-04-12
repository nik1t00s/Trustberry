from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import count_reviews, fetch_recent, init_db, insert_reviews
from .ml import METRICS_PATH, load_dataset, load_model, predict_one, train_model
from .schemas import PredictionOut, ReviewIn, TrainResponse


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".jl", ".txt"}

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
    session_loaded = dataset_name is not None
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "recent": fetch_recent(20),
            "result": result,
            "metrics": metrics if metrics is not None else saved_metrics,
            "message": message,
            "message_type": message_type,
            "dataset_name": dataset_name,
            "model_ready": saved_metrics is not None,
            "session_loaded": session_loaded,
            "reviews_total": count_reviews(),
        },
    )


def build_risk_badge(probability: float) -> dict[str, str]:
    if probability >= 0.75:
        return {"level": "Высокий риск", "tone": "danger"}
    if probability >= 0.4:
        return {"level": "Средний риск", "tone": "warning"}
    return {"level": "Низкий риск", "tone": "safe"}


def train_and_store(dataset_path: Path) -> tuple[dict[str, Any], int]:
    df = load_dataset(dataset_path)
    _, metrics = train_model(df)
    rows = df.to_dict(orient="records")
    insert_reviews(
        [{**row, "probability_fake": None, "predicted_label": None} for row in rows]
    )
    return metrics, len(rows)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render_index(request)


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
    sample_path = DATA_DIR / "sample_reviews.json"
    metrics, row_count = train_and_store(sample_path)
    return TrainResponse(
        message="Модель успешно обучена на демонстрационном наборе",
        train_size=metrics["train_size"],
        test_size=metrics["test_size"],
        metrics={
            **{k: v for k, v in metrics.items() if k not in {"classification_report"}},
            "stored_rows": row_count,
        },
    )


@app.post("/train-demo", response_class=HTMLResponse)
def train_demo_form(request: Request) -> HTMLResponse:
    sample_path = DATA_DIR / "sample_reviews.json"
    metrics, row_count = train_and_store(sample_path)
    return render_index(
        request,
        metrics={**metrics, "stored_rows": row_count},
        message="Демонстрационный набор успешно загружен, модель обучена.",
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
            message="Поддерживаются файлы CSV, JSON, JSONL, TXT(JSON Lines) и JSON.ZST.",
            message_type="error",
        )

    dst = UPLOADS_DIR / dataset.filename
    with dst.open("wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        metrics, row_count = train_and_store(dst)
    except Exception as exc:
        return render_index(
            request,
            message=f"Не удалось обработать датасет: {exc}",
            message_type="error",
            dataset_name=dataset.filename,
        )

    return render_index(
        request,
        metrics={**metrics, "stored_rows": row_count},
        message=f"Датасет загружен, очищен и использован для обучения модели. В БД сохранено записей: {row_count}.",
        message_type="success",
        dataset_name=dataset.filename,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/reviews")
def reviews() -> list[dict[str, Any]]:
    return fetch_recent(100)

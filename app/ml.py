from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency during local dev
    zstd = None


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "review_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
ZST_TMP_DIR = BASE_DIR / "data" / "tmp_zst"
ZST_TMP_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMNS = ["review_text", "seller_answer", "color"]
NUM_COLUMNS = ["rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"]
ALL_COLUMNS = ["product_id", "rating", "color", "review_text", "seller_answer", "fake_label"]


FAKE_HINTS = [
    "рекомендую", "берите", "супер", "огонь", "лучший", "идеально",
    "5 звезд", "всем советую", "пришло быстро", "продавец отличный",
    "за бонус", "за баллы", "за скидку"
]


LEGIT_HINTS = [
    "после недели", "после месяца", "использую", "заметил", "сравнил",
    "достоинства", "недостатки", "упаковка", "материал"
]


def read_json_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            data = [data]
    return pd.DataFrame(data)


def read_json_lines(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def read_zst_json(path: Path) -> pd.DataFrame:
    if zstd is None:
        raise ValueError(
            "Для файлов .json.zst требуется пакет zstandard. Добавьте его в окружение проекта."
        )

    temp_path = ZST_TMP_DIR / f"{uuid4().hex}_{path.name.removesuffix('.zst')}"
    try:
        with path.open("rb") as compressed_file:
            with temp_path.open("wb") as decompressed_file:
                stream_reader = zstd.ZstdDecompressor().stream_reader(compressed_file)
                try:
                    shutil.copyfileobj(stream_reader, decompressed_file)
                finally:
                    stream_reader.close()
        return detect_json_loader(temp_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                pass


def detect_json_loader(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return pd.DataFrame()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return read_json_lines(path)
    if isinstance(parsed, dict):
        if "data" in parsed and isinstance(parsed["data"], list):
            parsed = parsed["data"]
        else:
            parsed = [parsed]
    if isinstance(parsed, list):
        return pd.DataFrame(parsed)
    return read_json_lines(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "nmId": "product_id",
        "productValuation": "rating",
        "text": "review_text",
        "answer": "seller_answer",
        "label": "fake_label",
    }
    df = df.rename(columns=rename_map).copy()

    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[ALL_COLUMNS]



def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["review_text", "seller_answer", "color"]:
        work[col] = work[col].apply(clean_text)

    original_text = work["review_text"].fillna("")
    work["review_length"] = original_text.apply(lambda x: len(str(x).split()))
    work["exclamation_count"] = original_text.apply(lambda x: str(x).count("!"))
    work["digit_count"] = original_text.apply(lambda x: sum(ch.isdigit() for ch in str(x)))
    work["uppercase_ratio"] = original_text.apply(_uppercase_ratio)
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce")
    work["product_id"] = pd.to_numeric(work["product_id"], errors="coerce")
    return work



def _uppercase_ratio(text: Any) -> float:
    s = str(text)
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)



def weak_label_rule(review_text: str, rating: Any) -> int:
    text = clean_text(review_text)
    score = 0
    for hint in FAKE_HINTS:
        if hint in text:
            score += 1
    for hint in LEGIT_HINTS:
        if hint in text:
            score -= 1
    if len(text.split()) <= 3:
        score += 1
    if str(text).count("!") >= 2:
        score += 1
    try:
        if int(rating) == 5 and score >= 1:
            score += 1
    except Exception:
        pass
    return 1 if score >= 2 else 0



def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work["fake_label"].isna().all():
        work["fake_label"] = work.apply(
            lambda row: weak_label_rule(row.get("review_text"), row.get("rating")), axis=1
        )
    else:
        work["fake_label"] = pd.to_numeric(work["fake_label"], errors="coerce")
        unlabeled_mask = work["fake_label"].isna()
        work.loc[unlabeled_mask, "fake_label"] = work.loc[unlabeled_mask].apply(
            lambda row: weak_label_rule(row.get("review_text"), row.get("rating")), axis=1
        )
    work["fake_label"] = work["fake_label"].astype(int)
    return work



def load_dataset(path: Path) -> pd.DataFrame:
    suffixes = [part.lower() for part in path.suffixes]
    suffix = suffixes[-1] if suffixes else ""

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffixes[-2:] == [".json", ".zst"]:
        df = read_zst_json(path)
    elif suffix == ".json":
        df = detect_json_loader(path)
    elif suffix in {".jsonl", ".jl", ".txt"}:
        df = read_json_lines(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.name}")

    df = normalize_columns(df)
    df = ensure_labels(df)
    df = engineer_features(df)
    df = df[df["review_text"].astype(str).str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("После очистки не осталось непустых отзывов для обучения.")
    return df



class TextSelector:
    def __init__(self, colname: str):
        self.colname = colname

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.colname].fillna("")



def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "review_text_tfidf",
                Pipeline([
                    ("select", TextSelector("review_text")),
                    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                ]),
                ["review_text", "seller_answer", "color", "rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"],
            ),
            (
                "seller_answer_tfidf",
                Pipeline([
                    ("select", TextSelector("seller_answer")),
                    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ]),
                ["review_text", "seller_answer", "color", "rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"],
            ),
            (
                "color_ohe",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                ["color"],
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]),
                NUM_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])



def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    if len(df) < 4:
        raise ValueError("Для обучения нужно минимум 4 непустых отзыва.")
    if df["fake_label"].nunique() < 2:
        raise ValueError("Для обучения нужны отзывы минимум двух классов fake_label.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["fake_label"] if df["fake_label"].nunique() > 1 else None,
    )

    X_train = train_df[["review_text", "seller_answer", "color", "rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"]]
    y_train = train_df["fake_label"]
    X_test = test_df[["review_text", "seller_answer", "color", "rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"]]
    y_test = test_df["fake_label"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "classification_report": classification_report(y_test, preds, zero_division=0, output_dict=True),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "avg_probability_fake": round(float(probs.mean()), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return pipeline, metrics



def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Модель еще не обучена")
    return joblib.load(MODEL_PATH)



def predict_one(model: Pipeline, payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([
        {
            "product_id": payload.get("product_id"),
            "rating": payload.get("rating"),
            "color": payload.get("color"),
            "review_text": payload.get("review_text"),
            "seller_answer": payload.get("seller_answer"),
            "fake_label": 0,
        }
    ])
    df = engineer_features(df)
    X = df[["review_text", "seller_answer", "color", "rating", "review_length", "exclamation_count", "uppercase_ratio", "digit_count"]]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": pred,
        "label": "Мошеннический" if pred == 1 else "Легитимный",
        "probability_fake": round(prob, 4),
    }

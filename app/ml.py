from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataio import load_records, normalize_records


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "review_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

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
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(normalize_records(load_records(path)))

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


def predict_dataset(model: Pipeline, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(
        [
            {
                "product_id": row.get("nmId") or row.get("product_id"),
                "rating": row.get("productValuation") or row.get("rating"),
                "color": row.get("color"),
                "review_text": row.get("text") or row.get("review_text"),
                "seller_answer": row.get("answer") or row.get("seller_answer"),
                "fake_label": row.get("fake_label", 0) or 0,
            }
            for row in rows
        ]
    )
    df = engineer_features(df)
    X = df[
        [
            "review_text",
            "seller_answer",
            "color",
            "rating",
            "review_length",
            "exclamation_count",
            "uppercase_ratio",
            "digit_count",
        ]
    ]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return [
        {
            **row,
            "predicted_label": int(pred),
            "predicted_proba": round(float(prob), 4),
        }
        for row, pred, prob in zip(rows, predictions, probabilities, strict=False)
    ]


FEATURE_COLUMNS = [
    "review_text",
    "seller_answer",
    "color",
    "rating",
    "review_length",
    "exclamation_count",
    "uppercase_ratio",
    "digit_count",
]
MODEL_VERSION_PREFIX = "review_model"


def build_model_version() -> str:
    return f"{MODEL_VERSION_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_model_version() -> str:
    if METRICS_PATH.exists():
        try:
            payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            version = payload.get("model_version")
            if version:
                return str(version)
        except (json.JSONDecodeError, OSError):
            pass
    return f"{MODEL_VERSION_PREFIX}_untrained"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "nmId": "product_id",
        "productValuation": "rating",
        "text": "review_text",
        "answer": "seller_answer",
        "label": "fake_label",
    }
    work = df.rename(columns=rename_map).copy()
    for column in ALL_COLUMNS:
        if column not in work.columns:
            work[column] = None
    return work[ALL_COLUMNS]


def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ!?.,\s-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _uppercase_ratio(text: Any) -> float:
    source = str(text or "")
    letters = [char for char in source if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_columns(df)
    raw_text = work["review_text"].fillna("")
    raw_answer = work["seller_answer"].fillna("")

    work["review_text"] = raw_text.apply(clean_text)
    work["seller_answer"] = raw_answer.apply(clean_text)
    work["color"] = work["color"].fillna("").astype(str).str.strip().str.lower()
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce")
    work["product_id"] = pd.to_numeric(work["product_id"], errors="coerce")
    work["review_length"] = raw_text.apply(lambda value: len(str(value).split()))
    work["exclamation_count"] = raw_text.apply(lambda value: str(value).count("!"))
    work["digit_count"] = raw_text.apply(lambda value: sum(char.isdigit() for char in str(value)))
    work["uppercase_ratio"] = raw_text.apply(_uppercase_ratio)
    return work


def build_pipeline() -> Pipeline:
    feature_source = FEATURE_COLUMNS
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "review_text_tfidf",
                Pipeline(
                    [
                        ("select", TextSelector("review_text")),
                        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                    ]
                ),
                feature_source,
            ),
            (
                "seller_answer_tfidf",
                Pipeline(
                    [
                        ("select", TextSelector("seller_answer")),
                        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                    ]
                ),
                feature_source,
            ),
            (
                "color_ohe",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["color"],
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUM_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def load_training_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = engineer_features(df)
    df["fake_label"] = pd.to_numeric(df["fake_label"], errors="coerce").astype(int)
    df = df[df["review_text"].astype(str).str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("После очистки не осталось непустых отзывов для обучения.")
    return df


def load_prediction_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = engineer_features(df)
    return df


def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    if len(df) < 4:
        raise ValueError("Для обучения нужно минимум 4 размеченные записи.")
    if df["fake_label"].nunique() < 2:
        raise ValueError("Для обучения нужны записи обоих классов: fake_label = 0 и fake_label = 1.")

    positive_rate = float(df["fake_label"].mean())
    class_balance = {
        "label_0": int((df["fake_label"] == 0).sum()),
        "label_1": int((df["fake_label"] == 1).sum()),
    }
    stratify = df["fake_label"] if min(class_balance.values()) > 1 else None
    test_size = 0.25 if len(df) >= 8 else 0.5

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["fake_label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["fake_label"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    roc_auc: float | None
    try:
        roc_auc = round(float(roc_auc_score(y_test, probabilities)), 4)
    except ValueError:
        roc_auc = None

    confusion = confusion_matrix(y_test, predictions, labels=[0, 1])
    model_version = build_model_version()
    metrics = {
        "model_version": model_version,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion.tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "total_records": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "positive_rate": round(positive_rate, 4),
        "class_balance": class_balance,
        "avg_probability_fake": round(float(probabilities.mean()), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    return pipeline, metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Обученная модель не найдена. Сначала выполните обучение.")
    return joblib.load(MODEL_PATH)


def predict_one(model: Pipeline, payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame(
        [
            {
                "product_id": payload.get("product_id"),
                "rating": payload.get("rating"),
                "color": payload.get("color"),
                "review_text": payload.get("review_text"),
                "seller_answer": payload.get("seller_answer"),
                "fake_label": 0,
            }
        ]
    )
    df = engineer_features(df)
    prediction = int(model.predict(df[FEATURE_COLUMNS])[0])
    probability = float(model.predict_proba(df[FEATURE_COLUMNS])[0, 1])
    return {
        "prediction": prediction,
        "label": "Мошеннический" if prediction == 1 else "Не мошеннический",
        "probability_fake": round(probability, 4),
        "model_version": get_model_version(),
    }


def predict_dataset(model: Pipeline, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = load_prediction_dataframe(rows)
    predictions = model.predict(df[FEATURE_COLUMNS])
    probabilities = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    return [
        {
            **row,
            "predicted_label": int(prediction),
            "predicted_proba": round(float(probability), 4),
        }
        for row, prediction, probability in zip(rows, predictions, probabilities, strict=False)
    ]


from copy import deepcopy
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def build_model_version(model_type: str) -> str:
    short_name = {
        "logistic_regression": "logreg",
        "xgboost": "xgboost",
        "catboost": "catboost",
    }.get(model_type, model_type)
    return f"{MODEL_VERSION_PREFIX}_{short_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_metrics_payload() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_model_version() -> str:
    payload = load_metrics_payload()
    if payload:
        version = payload.get("summary", payload).get("model_version")
        if version:
            return str(version)
    return f"{MODEL_VERSION_PREFIX}_untrained"


def get_model_type() -> str:
    payload = load_metrics_payload()
    if payload:
        model_type = payload.get("summary", payload).get("model_type")
        if model_type:
            return str(model_type)
    return "untrained"


def get_available_model_options() -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "label": MODEL_LABELS["logistic_regression"],
            "available": True,
            "reason": "",
        },
        "xgboost": {
            "label": MODEL_LABELS["xgboost"],
            "available": XGBClassifier is not None,
            "reason": "" if XGBClassifier is not None else "Библиотека xgboost не установлена.",
        },
        "catboost": {
            "label": MODEL_LABELS["catboost"],
            "available": CatBoostClassifier is not None,
            "reason": "" if CatBoostClassifier is not None else "Библиотека catboost не установлена.",
        },
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "review_text_tfidf",
                Pipeline(
                    [
                        ("select", TextSelector("review_text")),
                        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                    ]
                ),
                FEATURE_COLUMNS,
            ),
            (
                "seller_answer_tfidf",
                Pipeline(
                    [
                        ("select", TextSelector("seller_answer")),
                        ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                    ]
                ),
                FEATURE_COLUMNS,
            ),
            (
                "color_ohe",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["color"],
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUM_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def split_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    if len(df) < 4:
        raise ValueError("Для обучения нужно минимум 4 размеченные записи.")
    if df["fake_label"].nunique() < 2:
        raise ValueError("Для обучения нужны записи обоих классов: fake_label = 0 и fake_label = 1.")

    positive_rate = float(df["fake_label"].mean())
    class_balance = {
        "label_0": int((df["fake_label"] == 0).sum()),
        "label_1": int((df["fake_label"] == 1).sum()),
    }
    stratify = df["fake_label"] if min(class_balance.values()) > 1 else None
    test_size = 0.25 if len(df) >= 8 else 0.5

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    split_meta = {
        "total_records": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "positive_rate": round(positive_rate, 4),
        "class_balance": class_balance,
    }
    return train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS], train_df["fake_label"], test_df["fake_label"], split_meta


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        return 1.0
    return round(negatives / positives, 4)


def create_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def create_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    if XGBClassifier is None:
        raise ValueError("XGBoost недоступен: библиотека xgboost не установлена.")
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                ),
            ),
        ]
    )


def create_catboost_pipeline(scale_pos_weight: float) -> Pipeline:
    if CatBoostClassifier is None:
        raise ValueError("CatBoost недоступен: библиотека catboost не установлена.")
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="F1",
                    random_state=42,
                    verbose=False,
                    scale_pos_weight=scale_pos_weight,
                    depth=6,
                    learning_rate=0.08,
                    iterations=250,
                ),
            ),
        ]
    )


def maybe_tune_xgboost(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, *, tuned: bool) -> Pipeline:
    if not tuned:
        return pipeline
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "classifier__n_estimators": [150, 250, 350],
            "classifier__max_depth": [4, 6, 8],
            "classifier__learning_rate": [0.03, 0.08, 0.12],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
        },
        n_iter=8,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_type: str,
    split_meta: dict[str, Any],
    source_file: str,
    tuned: bool,
) -> tuple[Pipeline, dict[str, Any]]:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    try:
        roc_auc = round(float(roc_auc_score(y_test, probabilities)), 4)
    except ValueError:
        roc_auc = None

    return pipeline, {
        "model_type": model_type,
        "model_label": MODEL_LABELS.get(model_type, model_type),
        "model_version": build_model_version(model_type),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": source_file,
        "tuned": tuned,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "avg_probability_fake": round(float(probabilities.mean()), 4),
        **split_meta,
    }


def train_single_model(
    df: pd.DataFrame,
    *,
    model_type: str,
    source_file: str,
    training_speed: str = "fast",
) -> tuple[Pipeline, dict[str, Any]]:
    X_train, X_test, y_train, y_test, split_meta = split_training_data(df)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    tuned = training_speed == "tuned"

    if model_type == "logistic_regression":
        pipeline = create_logistic_pipeline()
    elif model_type == "xgboost":
        pipeline = create_xgboost_pipeline(scale_pos_weight)
        pipeline = maybe_tune_xgboost(pipeline, X_train, y_train, tuned=tuned)
    elif model_type == "catboost":
        if tuned:
            raise ValueError("Для CatBoost режим подбора параметров пока не реализован.")
        pipeline = create_catboost_pipeline(scale_pos_weight)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    return evaluate_pipeline(
        pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type=model_type,
        split_meta=split_meta,
        source_file=source_file,
        tuned=tuned,
    )


def build_candidate_model_list() -> list[str]:
    candidates = ["logistic_regression"]
    if XGBClassifier is not None:
        candidates.append("xgboost")
    if CatBoostClassifier is not None:
        candidates.append("catboost")
    return candidates


def select_best_model(results: list[tuple[str, Pipeline, dict[str, Any]]]) -> tuple[str, Pipeline, dict[str, Any]]:
    return max(
        results,
        key=lambda item: (float(item[2]["f1"]), float(item[2]["roc_auc"] or 0.0)),
    )


def train_and_compare_models(
    df: pd.DataFrame,
    *,
    source_file: str,
    training_speed: str = "fast",
) -> tuple[Pipeline, dict[str, Any]]:
    results: list[tuple[str, Pipeline, dict[str, Any]]] = []
    unavailable_models: dict[str, str] = {}

    for model_type in build_candidate_model_list():
        try:
            pipeline, metrics = train_single_model(
                df,
                model_type=model_type,
                source_file=source_file,
                training_speed=training_speed,
            )
            results.append((model_type, pipeline, metrics))
        except ValueError as exc:
            unavailable_models[model_type] = str(exc)

    if len(results) < 2:
        raise ValueError("Для сравнения моделей нужна минимум еще одна доступная модель кроме Logistic Regression.")

    winner_model_type, winner_pipeline, winner_metrics = select_best_model(results)
    return winner_pipeline, {
        "summary": winner_metrics,
        "models": {model_type: metrics for model_type, _, metrics in results},
        "comparison": {
            "comparison_metrics": [
                {
                    "model_type": model_type,
                    "model_label": metrics["model_label"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                }
                for model_type, _, metrics in results
            ],
            "winner_model_type": winner_model_type,
            "winner_model_label": winner_metrics["model_label"],
            "criterion_used": "f1_then_roc_auc",
            "unavailable_models": unavailable_models,
        },
        "active_model": {
            "model_type": winner_metrics["model_type"],
            "model_version": winner_metrics["model_version"],
            "trained_at": winner_metrics["trained_at"],
            "source_file": winner_metrics["source_file"],
        },
    }


def train_model(
    df: pd.DataFrame,
    *,
    source_file: str,
    mode: str = "single",
    model_type: str = "logistic_regression",
    training_speed: str = "fast",
) -> tuple[Pipeline, dict[str, Any]]:
    if mode == "compare":
        return train_and_compare_models(df, source_file=source_file, training_speed=training_speed)

    pipeline, metrics = train_single_model(
        df,
        model_type=model_type,
        source_file=source_file,
        training_speed=training_speed,
    )
    return pipeline, {
        "summary": metrics,
        "models": {model_type: deepcopy(metrics)},
        "comparison": None,
        "active_model": {
            "model_type": metrics["model_type"],
            "model_version": metrics["model_version"],
            "trained_at": metrics["trained_at"],
            "source_file": metrics["source_file"],
        },
    }


def save_trained_model(pipeline: Pipeline, metrics_payload: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def predict_one(model: Pipeline, payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame(
        [
            {
                "product_id": payload.get("product_id"),
                "rating": payload.get("rating"),
                "color": payload.get("color"),
                "review_text": payload.get("review_text"),
                "seller_answer": payload.get("seller_answer"),
                "fake_label": 0,
            }
        ]
    )
    df = engineer_features(df)
    prediction = int(model.predict(df[FEATURE_COLUMNS])[0])
    probability = float(model.predict_proba(df[FEATURE_COLUMNS])[0, 1])
    return {
        "prediction": prediction,
        "label": "Мошеннический" if prediction == 1 else "Не мошеннический",
        "probability_fake": round(probability, 4),
        "model_version": get_model_version(),
        "model_type": get_model_type(),
    }

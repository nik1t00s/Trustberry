from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.dataio import load_records, validate_training_records
from app.ml import (
    FEATURE_COLUMNS,
    build_preprocessor,
    compute_scale_pos_weight,
    load_training_dataframe,
)

DATASET_PATH = BASE_DIR / "data" / "labeled_records.jsonl"
OUTPUT_DIR = BASE_DIR / "docs" / "imbalance_analysis"
RESULTS_CSV_PATH = OUTPUT_DIR / "imbalance_results.csv"
REPORT_PATH = OUTPUT_DIR / "DISBALANCE.md"


def random_oversample(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train = X.copy()
    train["fake_label"] = y.to_numpy()
    majority = train[train["fake_label"] == 0]
    minority = train[train["fake_label"] == 1]

    if minority.empty or majority.empty or len(minority) >= len(majority):
        return X, y

    minority_upsampled = minority.sample(
        n=len(majority),
        replace=True,
        random_state=42,
    )
    balanced = pd.concat([majority, minority_upsampled], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
    y_balanced = balanced["fake_label"].astype(int)
    X_balanced = balanced.drop(columns=["fake_label"])
    return X_balanced, y_balanced


def build_logistic_pipeline(*, class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    C=1.0,
                    solver="liblinear",
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    scale_pos_weight=scale_pos_weight,
                    reg_lambda=2.0,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_catboost_pipeline(class_weights: list[float]) -> Pipeline:
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
                    depth=6,
                    learning_rate=0.08,
                    iterations=250,
                    class_weights=class_weights,
                ),
            ),
        ]
    )


def select_threshold_by_f1_positive(y_true: pd.Series, probabilities: Any) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    threshold = 0.10
    while threshold <= 0.900001:
        predictions = (probabilities >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, pos_label=1, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = round(threshold, 2)
        threshold += 0.05
    return best_threshold, round(best_score, 4)


def evaluate_predictions(
    y_true: pd.Series,
    predictions: Any,
    probabilities: Any | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision_class_1": round(float(precision_score(y_true, predictions, pos_label=1, zero_division=0)), 4),
        "recall_class_1": round(float(recall_score(y_true, predictions, pos_label=1, zero_division=0)), 4),
        "f1_class_1": round(float(f1_score(y_true, predictions, pos_label=1, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    if probabilities is None:
        result["roc_auc"] = None
        result["pr_auc"] = None
    else:
        result["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 4)
        result["pr_auc"] = round(float(average_precision_score(y_true, probabilities)), 4)
    return result


def save_confusion_matrix(confusion: list[list[int]], output_path: Path, title: str) -> None:
    matrix = pd.DataFrame(confusion, index=["0", "1"], columns=["0", "1"])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(matrix.values, cmap="Greys", vmin=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    max_value = matrix.values.max() if matrix.values.size else 0
    threshold = max_value / 2 if max_value else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "не рассчитано"
    return f"{float(value):.4f}"


def markdown_results_table(results_df: pd.DataFrame) -> str:
    headers = [
        "Подход",
        "Модель",
        "Порог классификации",
        "Accuracy",
        "Precision класса 1",
        "Recall класса 1",
        "F1 класса 1",
        "Macro-F1",
        "ROC-AUC",
        "PR-AUC",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in results_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["approach"]),
                    str(row["model"]),
                    format_metric(row["threshold"]),
                    format_metric(row["accuracy"]),
                    format_metric(row["precision_class_1"]),
                    format_metric(row["recall_class_1"]),
                    format_metric(row["f1_class_1"]),
                    format_metric(row["macro_f1"]),
                    format_metric(row["roc_auc"]),
                    format_metric(row["pr_auc"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def choose_preferred_row(results_df: pd.DataFrame) -> pd.Series:
    ranking = results_df.copy()
    ranking["roc_auc_rank"] = ranking["roc_auc"].fillna(-1.0)
    ranking["pr_auc_rank"] = ranking["pr_auc"].fillna(-1.0)
    ranking = ranking.sort_values(
        by=["macro_f1", "f1_class_1", "recall_class_1", "pr_auc_rank", "roc_auc_rank"],
        ascending=False,
    )
    return ranking.iloc[0]


def build_report(
    *,
    validation: dict[str, Any],
    class_counts: dict[int, int],
    split_sizes: dict[str, int],
    threshold_info: dict[str, Any],
    results_df: pd.DataFrame,
) -> str:
    total = int(sum(class_counts.values()))
    class_0 = int(class_counts.get(0, 0))
    class_1 = int(class_counts.get(1, 0))
    share_0 = class_0 / total if total else 0.0
    share_1 = class_1 / total if total else 0.0
    ratio = (class_0 / class_1) if class_1 else None
    preferred = choose_preferred_row(results_df)

    findings: list[str] = []
    for _, row in results_df.sort_values(by=["macro_f1", "f1_class_1"], ascending=False).iterrows():
        findings.append(
            (
                f"Подход «{row['approach']}» на модели {row['model']} обеспечил "
                f"Macro-F1 = {format_metric(row['macro_f1'])}, "
                f"F1 класса 1 = {format_metric(row['f1_class_1'])}, "
                f"Recall класса 1 = {format_metric(row['recall_class_1'])}, "
                f"PR-AUC = {format_metric(row['pr_auc'])}."
            )
        )

    report_lines = [
        "# Анализ дисбаланса классов в наборе данных",
        "",
        "## 1. Актуальность проблемы дисбаланса классов",
        "В задаче обнаружения мошеннических отзывов классы, как правило, распределены неравномерно: количество добросовестных отзывов существенно превышает количество подозрительных. В такой ситуации модель может демонстрировать высокую общую точность даже при неудовлетворительном качестве выявления мошеннических отзывов. Для рассматриваемой информационной системы это критично, поскольку пропуск объектов положительного класса снижает практическую ценность модуля автоматического контроля. По этой причине анализ методов компенсации дисбаланса классов представляет собой необходимый этап доработки модели.",
        "",
        "## 2. Распределение классов в исходном наборе данных",
        f"Для эксперимента использован файл `{DATASET_PATH.relative_to(BASE_DIR).as_posix()}`. После валидации и проверки дубликатов в наборе осталось {validation['valid_records']} записей.",
        "",
        f"- Количество объектов класса 0: {class_0}.",
        f"- Количество объектов класса 1: {class_1}.",
        f"- Доля класса 0: {share_0:.4f}.",
        f"- Доля класса 1: {share_1:.4f}.",
        f"- Соотношение классов 0:1: {ratio:.2f}:1." if ratio is not None else "- Соотношение классов 0:1 не рассчитано.",
        "",
        "## 3. Методика эксперимента",
        "Для исключения утечки данных использовалось стратифицированное разбиение на обучающую, валидационную и тестовую выборки. На первом шаге набор был разделен на train+validation и test в пропорции 80:20, после чего train+validation был дополнительно разделен на train и validation в пропорции 75:25. Итоговые размеры выборок составили:",
        "",
        f"- train: {split_sizes['train']} записей;",
        f"- validation: {split_sizes['validation']} записей;",
        f"- test: {split_sizes['test']} записей.",
        "",
        "Предобработка данных выполнялась средствами существующего проекта: текст отзыва преобразовывался с помощью TF-IDF по словам и символьным n-граммам, категориальный признак color кодировался методом One-Hot Encoding, числовые признаки масштабировались. Векторизатор обучался только на обучающей части в составе `Pipeline`, что исключало использование сведений из validation и test при построении признакового пространства.",
        "",
        "В экспериментах были проверены следующие подходы: базовая Logistic Regression без компенсации дисбаланса, Logistic Regression с `class_weight='balanced'`, Logistic Regression с подбором порога классификации по валидационной выборке, XGBoost с параметром `scale_pos_weight`, CatBoost с `class_weights`, а также случайный oversampling меньшинства только на обучающей части.",
        "",
        "Оценка качества выполнялась по метрикам Accuracy, Precision для класса 1, Recall для класса 1, F1-score для класса 1, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC и confusion matrix. Метрика Accuracy не использовалась как основная, поскольку при выраженном дисбалансе она может оставаться высокой за счет преобладающего класса 0 и не отражать качество обнаружения мошеннических отзывов.",
        "",
        "## 4. Проверенные подходы к обработке дисбаланса",
        "Базовый подход предполагал обучение Logistic Regression без специальных механизмов компенсации дисбаланса. Во втором варианте использовались взвешенные классы `class_weight='balanced'`, что увеличивает штраф за ошибки на объектах меньшинства. В третьем варианте для Logistic Regression подбирался порог классификации на validation-выборке по максимуму F1-score положительного класса; затем выбранный порог однократно применялся к test-выборке. Для XGBoost значение `scale_pos_weight` рассчитывалось как отношение числа объектов класса 0 к числу объектов класса 1 на обучающей части. Для CatBoost использовались веса классов `[1.0, scale_pos_weight]`. Дополнительно был реализован случайный oversampling класса 1 только на train-выборке без модификации validation и test.",
        "",
        f"В эксперименте с подбором порога оптимальным по validation-выборке оказался порог {threshold_info['threshold']:.2f}, при котором F1-score класса 1 на validation составил {threshold_info['validation_f1']:.4f}.",
        "",
        "## 5. Результаты экспериментов",
        markdown_results_table(results_df),
        "",
        "## 6. Анализ результатов",
    ]
    report_lines.extend([f"- {item}" for item in findings])
    report_lines.extend(
        [
            "",
            (
                f"По совокупности ключевых показателей предпочтительным оказался подход "
                f"«{preferred['approach']}» на модели {preferred['model']}, поскольку он обеспечил "
                f"наибольшее значение Macro-F1 = {format_metric(preferred['macro_f1'])}."
            ),
            (
                f"При этом значение F1-score класса 1 составило {format_metric(preferred['f1_class_1'])}, "
                f"Recall класса 1 — {format_metric(preferred['recall_class_1'])}, PR-AUC — {format_metric(preferred['pr_auc'])}."
            ),
            "Сопоставление Precision и Recall показывает наличие стандартного компромисса: методы, ориентированные на более активное выявление положительного класса, как правило, увеличивают Recall класса 1, однако могут сопровождаться снижением Precision за счет роста числа ложноположительных срабатываний. Следовательно, выбор рабочего режима должен учитывать допустимый баланс между пропуском мошеннических отзывов и числом ошибочных тревог.",
            "",
            "## 7. Выбранный подход для итоговой реализации",
            (
                f"С учетом проведенных расчетов целесообразно рекомендовать подход "
                f"«{preferred['approach']}» на модели {preferred['model']} как основной вариант для дальнейшего использования в системе. "
                f"Данный вывод основан на фактически рассчитанных значениях Macro-F1, F1-score класса 1, Recall класса 1 и PR-AUC. "
                f"Если в эксплуатационном сценарии приоритетом будет максимальное снижение пропусков положительного класса, "
                f"дополнительно допустимо рассматривать режим с более высоким Recall, даже при некотором ухудшении Precision."
            ),
            "",
            "## 8. Ограничения проведенного эксперимента",
            "- Результаты зависят от качества ручной разметки и возможной субъективности при присвоении меток.",
            "- Объем размеченных данных ограничивает устойчивость выводов и может влиять на переносимость результатов.",
            "- Итоговые показатели чувствительны к составу обучающей выборки даже при стратифицированном разбиении.",
            "- На неоднозначных, кратких или контекстно-зависимых отзывах возможны ошибки классификации.",
            "",
            "## 9. Текст для вставки в ВКР",
            "### Текст для раздела 4.1",
            (
                "При обучении модели обнаружения мошеннических отзывов была выявлена выраженная проблема дисбаланса классов. "
                f"После валидации размеченного набора данных было получено {class_0} отзывов класса 0 и {class_1} отзывов класса 1, "
                f"что соответствует соотношению приблизительно {ratio:.2f}:1 в пользу отрицательного класса. "
                "При таком распределении использование одной лишь метрики accuracy не позволяет объективно оценить качество модели, "
                "поскольку высокий результат может достигаться за счет доминирующего класса. В связи с этим в исследовании основное внимание "
                "было уделено метрикам F1-score для положительного класса, Macro-F1, Recall класса 1 и PR-AUC, позволяющим корректнее оценивать "
                "способность модели выявлять подозрительные отзывы."
            ),
            "",
            "### Текст для раздела 4.2",
            (
                "Для компенсации дисбаланса классов было проведено сравнение нескольких подходов на стратифицированном разбиении "
                "train/validation/test без утечки данных между этапами обучения и контроля качества. "
                "Были исследованы базовая Logistic Regression, Logistic Regression с `class_weight='balanced'`, "
                "Logistic Regression с подбором порога классификации по валидационной выборке, XGBoost с `scale_pos_weight`, "
                "CatBoost с весами классов и случайный oversampling меньшинства на обучающей части. "
                f"Наилучшие результаты по совокупности приоритетных метрик продемонстрировал подход «{preferred['approach']}» "
                f"на модели {preferred['model']}, обеспечивший Macro-F1 = {format_metric(preferred['macro_f1'])}, "
                f"F1-score класса 1 = {format_metric(preferred['f1_class_1'])}, Recall класса 1 = {format_metric(preferred['recall_class_1'])} "
                f"и PR-AUC = {format_metric(preferred['pr_auc'])}. Полученные результаты подтверждают, что учет дисбаланса классов "
                "является практически значимым фактором для повышения качества обнаружения мошеннических отзывов."
            ),
        ]
    )
    return "\n".join(report_lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(DATASET_PATH)
    validated_records, validation = validate_training_records(
        raw_records,
        source_file=DATASET_PATH.name,
        deduplicate=True,
    )
    df = load_training_dataframe(validated_records)

    class_counts = {
        0: int((df["fake_label"] == 0).sum()),
        1: int((df["fake_label"] == 1).sum()),
    }

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["fake_label"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=42,
        stratify=train_val_df["fake_label"],
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["fake_label"].astype(int)
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["fake_label"].astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["fake_label"].astype(int)

    split_sizes = {
        "train": int(len(train_df)),
        "validation": int(len(val_df)),
        "test": int(len(test_df)),
    }
    scale_pos_weight = compute_scale_pos_weight(y_train)
    results: list[dict[str, Any]] = []

    baseline_pipeline = build_logistic_pipeline(class_weight=None)
    baseline_pipeline.fit(X_train, y_train)
    baseline_probabilities = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_predictions = (baseline_probabilities >= 0.5).astype(int)
    baseline_metrics = evaluate_predictions(y_test, baseline_predictions, baseline_probabilities)
    results.append(
        {
            "approach": "Baseline",
            "model": "Logistic Regression",
            "threshold": 0.5,
            **baseline_metrics,
        }
    )

    balanced_pipeline = build_logistic_pipeline(class_weight="balanced")
    balanced_pipeline.fit(X_train, y_train)
    balanced_probabilities = balanced_pipeline.predict_proba(X_test)[:, 1]
    balanced_predictions = (balanced_probabilities >= 0.5).astype(int)
    balanced_metrics = evaluate_predictions(y_test, balanced_predictions, balanced_probabilities)
    results.append(
        {
            "approach": "Class Weight Balanced",
            "model": "Logistic Regression",
            "threshold": 0.5,
            **balanced_metrics,
        }
    )

    threshold_pipeline = build_logistic_pipeline(class_weight=None)
    threshold_pipeline.fit(X_train, y_train)
    validation_probabilities = threshold_pipeline.predict_proba(X_val)[:, 1]
    best_threshold, validation_f1 = select_threshold_by_f1_positive(y_val, validation_probabilities)
    threshold_probabilities = threshold_pipeline.predict_proba(X_test)[:, 1]
    threshold_predictions = (threshold_probabilities >= best_threshold).astype(int)
    threshold_metrics = evaluate_predictions(y_test, threshold_predictions, threshold_probabilities)
    results.append(
        {
            "approach": "Threshold Tuning",
            "model": "Logistic Regression",
            "threshold": best_threshold,
            **threshold_metrics,
        }
    )

    xgboost_pipeline = build_xgboost_pipeline(scale_pos_weight)
    xgboost_pipeline.fit(X_train, y_train)
    xgboost_probabilities = xgboost_pipeline.predict_proba(X_test)[:, 1]
    xgboost_predictions = (xgboost_probabilities >= 0.5).astype(int)
    xgboost_metrics = evaluate_predictions(y_test, xgboost_predictions, xgboost_probabilities)
    results.append(
        {
            "approach": "Scale Pos Weight",
            "model": "XGBoost",
            "threshold": 0.5,
            **xgboost_metrics,
        }
    )

    catboost_pipeline = build_catboost_pipeline([1.0, scale_pos_weight])
    catboost_pipeline.fit(X_train, y_train)
    catboost_probabilities = catboost_pipeline.predict_proba(X_test)[:, 1]
    catboost_predictions = (catboost_probabilities >= 0.5).astype(int)
    catboost_metrics = evaluate_predictions(y_test, catboost_predictions, catboost_probabilities)
    results.append(
        {
            "approach": "Class Weights",
            "model": "CatBoost",
            "threshold": 0.5,
            **catboost_metrics,
        }
    )

    X_over, y_over = random_oversample(X_train, y_train)
    oversampling_pipeline = build_logistic_pipeline(class_weight=None)
    oversampling_pipeline.fit(X_over, y_over)
    oversampling_probabilities = oversampling_pipeline.predict_proba(X_test)[:, 1]
    oversampling_predictions = (oversampling_probabilities >= 0.5).astype(int)
    oversampling_metrics = evaluate_predictions(y_test, oversampling_predictions, oversampling_probabilities)
    results.append(
        {
            "approach": "Random Oversampling",
            "model": "Logistic Regression",
            "threshold": 0.5,
            **oversampling_metrics,
        }
    )

    results_df = pd.DataFrame(results)
    results_df["confusion_matrix"] = results_df["confusion_matrix"].apply(json.dumps)
    results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")

    for row in results:
        slug = row["approach"].lower().replace(" ", "_")
        if slug == "class_weight_balanced":
            file_name = "confusion_balanced.png"
        elif slug == "baseline":
            file_name = "confusion_baseline.png"
        elif slug == "threshold_tuning":
            file_name = "confusion_threshold.png"
        else:
            file_name = f"confusion_{slug}.png"
        save_confusion_matrix(
            row["confusion_matrix"],
            OUTPUT_DIR / file_name,
            f"{row['approach']} ({row['model']})",
        )

    report = build_report(
        validation=validation,
        class_counts=class_counts,
        split_sizes=split_sizes,
        threshold_info={"threshold": best_threshold, "validation_f1": validation_f1},
        results_df=pd.DataFrame(results),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

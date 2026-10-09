"""训练 A2 乳腺癌二分类模型并保存真实实验结果。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.experiment import build_models, evaluate_predictions, load_dataset, split_dataset


RANDOM_STATE = 2026
MODEL_ORDER = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "Gradient Boosting",
    "MLP Neural Network",
]
TUNED_MODELS = {"Random Forest", "Gradient Boosting", "MLP Neural Network"}


def parameter_grids(fast_mode: bool) -> dict[str, dict[str, list]]:
    """返回三个需要调参模型的有限参数网格。"""
    if fast_mode:
        return {
            "Random Forest": {"n_estimators": [200], "max_depth": [None], "min_samples_leaf": [1]},
            "Gradient Boosting": {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [2]},
            "MLP Neural Network": {
                "mlpclassifier__hidden_layer_sizes": [(64, 32)],
                "mlpclassifier__alpha": [0.0001],
                "mlpclassifier__learning_rate_init": [0.001],
            },
        }
    return {
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 6],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        "MLP Neural Network": {
            "mlpclassifier__hidden_layer_sizes": [(32,), (64, 32)],
            "mlpclassifier__alpha": [0.0001, 0.001],
            "mlpclassifier__learning_rate_init": [0.001, 0.01],
        },
    }


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _save_tuning_rows(model_name: str, search: GridSearchCV) -> list[dict]:
    rows = []
    results = search.cv_results_
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "Model": model_name,
                "Parameters": json.dumps(params, ensure_ascii=False, default=_json_safe),
                "Mean CV F1": results["mean_test_score"][index],
                "Std CV F1": results["std_test_score"][index],
                "Rank": results["rank_test_score"][index],
            }
        )
    return rows


def run_experiment(data_path: str | Path, output_dir: str | Path, fast_mode: bool = False) -> dict:
    """在训练集完成交叉验证与调参，并对独立测试集作一次最终评价。"""
    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_dataset(features, target, RANDOM_STATE)
    models = build_models(RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    grids = parameter_grids(fast_mode)

    cv_rows: list[dict] = []
    tuning_rows: list[dict] = []
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    parameters: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for model_name in MODEL_ORDER:
        base_model = clone(models[model_name])
        if model_name in TUNED_MODELS:
            search = GridSearchCV(
                base_model,
                param_grid=grids[model_name],
                scoring="f1",
                cv=cv,
                n_jobs=1,
                refit=True,
                return_train_score=False,
            )
            fit_start = time.perf_counter()
            search.fit(x_train, y_train)
            elapsed = time.perf_counter() - fit_start
            final_model = search.best_estimator_
            parameters[model_name] = {
                "selection": "GridSearchCV on training set only",
                "best_parameters": search.best_params_,
                "best_cv_f1": search.best_score_,
            }
            tuning_rows.extend(_save_tuning_rows(model_name, search))
        else:
            final_model = base_model
            fit_start = time.perf_counter()
            final_model.fit(x_train, y_train)
            elapsed = time.perf_counter() - fit_start
            parameters[model_name] = {"selection": "default parameters", "parameters": final_model.get_params()}

        cv_scores = cross_validate(final_model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append(
            {
                "Model": model_name,
                "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
                "CV Precision Mean": cv_scores["test_precision"].mean(),
                "CV Recall Mean": cv_scores["test_recall"].mean(),
                "CV F1 Mean": cv_scores["test_f1"].mean(),
                "CV ROC-AUC Mean": cv_scores["test_roc_auc"].mean(),
                "CV F1 Std": cv_scores["test_f1"].std(),
                "Selection Method": (
                    "5-fold GridSearchCV (F1) + 5-fold evaluation"
                    if model_name in TUNED_MODELS
                    else "5-fold cross validation"
                ),
            }
        )

        predictions = final_model.predict(x_test)
        probabilities = final_model.predict_proba(x_test)[:, 1]
        metrics = evaluate_predictions(y_test, predictions, probabilities)
        metrics.update({"Model": model_name, "Training Time Seconds": elapsed})
        metric_rows.append(metrics)
        fitted_models[model_name] = final_model
        for sample_id, actual, predicted, probability in zip(x_test.index, y_test, predictions, probabilities):
            prediction_rows.append(
                {
                    "Model": model_name,
                    "Sample Index": int(sample_id),
                    "Actual": int(actual),
                    "Predicted": int(predicted),
                    "Positive Probability": float(probability),
                }
            )

    pd.DataFrame(cv_rows).to_csv(table_dir / "cv_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tuning_rows).to_csv(table_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(table_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(table_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    with (table_dir / "model_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(parameters, file, ensure_ascii=False, indent=2, default=_json_safe)

    return {
        "output_dir": output_root,
        "metrics": pd.DataFrame(metric_rows),
        "cv_results": pd.DataFrame(cv_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "fitted_models": fitted_models,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    course_root = PROJECT_DIR.parent
    source_data = course_root / "A类题" / "A2_classification_breast_cancer.csv"
    result_root = PROJECT_DIR / "results"
    result = run_experiment(source_data, result_root)
    print("A2 实验已完成。")
    print(result["metrics"].sort_values("F1", ascending=False).to_string(index=False))

"""根据 A2 的真实结果表生成报告证据图片。"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, learning_curve


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.experiment import build_models, load_dataset, split_dataset
from src.run_experiment import RANDOM_STATE


sns.set_theme(style="whitegrid")
windows_font = Path("C:/Windows/Fonts/msyh.ttc")
if windows_font.is_file():
    font_manager.fontManager.addfont(str(windows_font))
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_figures(
    data_path: str | Path, table_dir: str | Path, figure_dir: str | Path
) -> dict[str, Path]:
    """由真实数据和结果表生成五张可插入报告的 PNG 图。"""
    source_data = Path(data_path)
    table_root = Path(table_dir)
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    for prior_figure in output.glob("*.png"):
        prior_figure.unlink()

    metrics = pd.read_csv(table_root / "test_metrics.csv")
    cv_results = pd.read_csv(table_root / "cv_results.csv")
    predictions = pd.read_csv(table_root / "predictions.csv")
    raw_data = pd.read_csv(source_data)
    best_model = metrics.sort_values(["F1", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    paths = {
        "class_distribution": output / "class_distribution.png",
        "cv_f1_comparison": output / "cv_f1_comparison.png",
        "best_model_confusion_matrix": output / "best_model_confusion_matrix.png",
        "roc_curves": output / "roc_curves.png",
        "learning_curve": output / "learning_curve.png",
    }

    counts = raw_data["target"].value_counts().sort_index()
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(["恶性（0）", "良性（1）"], counts.values, color=["#C65D5D", "#4C8C6C"])
    plt.title("图1 A2 数据集目标类别分布")
    plt.ylabel("样本数")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center")
    _save_current_figure(paths["class_distribution"])

    ordered_cv = cv_results.set_index("Model").loc[metrics["Model"]].reset_index()
    plt.figure(figsize=(9, 5))
    bars = plt.bar(ordered_cv["Model"], ordered_cv["CV F1 Mean"], color="#4D79A6")
    plt.ylim(0.85, 1.0)
    plt.title("图2 五种模型的 5 折交叉验证 F1 对比")
    plt.ylabel("平均 F1")
    plt.xticks(rotation=18, ha="right")
    for bar, value in zip(bars, ordered_cv["CV F1 Mean"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", fontsize=9)
    _save_current_figure(paths["cv_f1_comparison"])

    best_predictions = predictions[predictions["Model"] == best_model]
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        best_predictions["Actual"],
        best_predictions["Predicted"],
        display_labels=["恶性（0）", "良性（1）"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"图3 本次固定测试集 F1 最高模型 {best_model} 的混淆矩阵")
    _save_current_figure(paths["best_model_confusion_matrix"])

    plt.figure(figsize=(7, 6))
    for model_name, frame in predictions.groupby("Model", sort=False):
        RocCurveDisplay.from_predictions(
            frame["Actual"], frame["Positive Probability"], name=model_name, ax=plt.gca()
        )
    plt.title("图4 五种模型的测试集 ROC 曲线")
    _save_current_figure(paths["roc_curves"])

    features, target = load_dataset(source_data)
    x_train, _, y_train, _ = split_dataset(features, target, RANDOM_STATE)
    estimator = build_models(RANDOM_STATE)[best_model]
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator,
        x_train,
        y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        n_jobs=1,
    )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="训练 F1", color="#D77A61")
    plt.plot(train_sizes, validation_scores.mean(axis=1), "o-", label="验证 F1", color="#4D79A6")
    plt.fill_between(
        train_sizes,
        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
        color="#4D79A6",
        alpha=0.16,
    )
    plt.title(f"图5 本次固定测试集 F1 最高模型 {best_model} 的学习曲线")
    plt.xlabel("训练样本数")
    plt.ylabel("F1")
    plt.ylim(0.85, 1.01)
    plt.legend()
    _save_current_figure(paths["learning_curve"])

    return paths


if __name__ == "__main__":
    course_root = PROJECT_DIR.parent
    figure_paths = generate_figures(
        course_root / "A类题" / "A2_classification_breast_cancer.csv",
        PROJECT_DIR / "results" / "tables",
        PROJECT_DIR / "results" / "figures",
    )
    print("已生成证据图片：")
    for path in figure_paths.values():
        print(path.name)

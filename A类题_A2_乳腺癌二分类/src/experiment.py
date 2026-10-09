"""A2 乳腺癌二分类的数据与评估基础函数。"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """读取 A2 CSV，移除样本标识字段并返回特征矩阵和二分类目标。"""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    dataframe = pd.read_csv(data_path)
    if "target" not in dataframe.columns:
        raise ValueError("数据必须包含 target 目标字段。")
    if "sample_id" not in dataframe.columns:
        raise ValueError("数据必须包含 sample_id 字段，便于明确将其排除在特征之外。")

    features = dataframe.drop(columns=["sample_id", "target"])
    target = dataframe["target"].copy()
    if target.nunique() != 2:
        raise ValueError("A2 任务要求二分类 target 标签。")
    return features, target


def split_dataset(
    features: pd.DataFrame, target: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """以固定随机种子执行 80%/20% 的分层训练测试划分。"""
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )


def build_models(random_state: int) -> dict[str, object]:
    """构造五个在统一划分上比较的候选分类模型。"""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "MLP Neural Network": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    }


def evaluate_predictions(y_true, y_pred, y_score) -> dict[str, float]:
    """返回 A2 任务统一使用的分类指标。"""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
    }

from pathlib import Path
import sys
import unittest
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT.parents[0] / "A类题" / "A2_classification_breast_cancer.csv"


class DatasetLoadingTest(unittest.TestCase):
    def test_load_dataset_excludes_sample_id_and_preserves_binary_features(self):
        from src.experiment import load_dataset

        features, target = load_dataset(DATA_PATH)

        self.assertNotIn("sample_id", features.columns)
        self.assertEqual(features.shape, (569, 30))
        self.assertEqual(set(target.unique()), {0, 1})

    def test_split_models_and_metrics_have_required_structure(self):
        from src.experiment import (
            build_models,
            evaluate_predictions,
            load_dataset,
            split_dataset,
        )

        features, target = load_dataset(DATA_PATH)
        x_train, x_test, y_train, y_test = split_dataset(features, target, 2026)
        self.assertEqual((len(x_train), len(x_test)), (455, 114))
        self.assertEqual(set(y_train.unique()), {0, 1})
        self.assertEqual(set(y_test.unique()), {0, 1})

        expected_models = {
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "Gradient Boosting",
            "MLP Neural Network",
        }
        self.assertEqual(set(build_models(2026)), expected_models)

        metrics = evaluate_predictions(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0.1, 0.9, 0.4, 0.2]),
        )
        self.assertEqual(
            set(metrics),
            {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC"},
        )

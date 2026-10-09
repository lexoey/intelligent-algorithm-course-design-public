from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH = PROJECT_ROOT.parents[0] / "A类题" / "A2_classification_breast_cancer.csv"


class ExperimentOutputTest(unittest.TestCase):
    def test_run_experiment_creates_complete_nonempty_result_tables(self):
        from src.run_experiment import run_experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            run_experiment(DATA_PATH, output_dir, fast_mode=True)

            table_dir = output_dir / "tables"
            expected_files = {
                "cv_results.csv",
                "tuning_results.csv",
                "test_metrics.csv",
                "predictions.csv",
                "model_parameters.json",
            }
            self.assertTrue(expected_files.issubset({item.name for item in table_dir.iterdir()}))

            metrics = pd.read_csv(table_dir / "test_metrics.csv")
            self.assertEqual(len(metrics), 5)
            self.assertTrue(
                {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Training Time Seconds"}
                .issubset(metrics.columns)
            )
            self.assertFalse(metrics.isna().any().any())

            cv_results = pd.read_csv(table_dir / "cv_results.csv")
            self.assertEqual(len(cv_results), 5)
            self.assertFalse(
                cv_results[
                    [
                        "CV Accuracy Mean",
                        "CV Precision Mean",
                        "CV Recall Mean",
                        "CV F1 Mean",
                        "CV ROC-AUC Mean",
                    ]
                ].isna().any().any()
            )

    def test_generate_figures_creates_five_nonempty_png_files(self):
        from src.run_experiment import run_experiment
        from src.plot_results import generate_figures

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            figure_dir = Path(temp_dir) / "figures"
            run_experiment(DATA_PATH, output_dir, fast_mode=True)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                generate_figures(DATA_PATH, output_dir / "tables", figure_dir)
            expected_files = {
                "class_distribution.png",
                "cv_f1_comparison.png",
                "best_model_confusion_matrix.png",
                "roc_curves.png",
                "learning_curve.png",
            }
            self.assertEqual(expected_files, {item.name for item in figure_dir.glob("*.png")})
            self.assertTrue(all(item.stat().st_size > 0 for item in figure_dir.glob("*.png")))
            self.assertFalse(any("Glyph" in str(item.message) for item in captured))

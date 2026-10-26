"""
Unit tests for 4 Model Families & Stratified Cross-Validation Tournament.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from data_agent.models.trainer import ModelTournamentTrainer

def test_four_model_families_tournament():
    X_arr, y_arr = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(6)])
    y = pd.Series(y_arr, name="target")

    trainer = ModelTournamentTrainer(n_splits=3, search_iterations=2, random_state=42)
    report = trainer.train_and_evaluate(X, y, task_type="binary_classification")

    # 1. Verify all 4 model families competed in the tournament
    expected_models = {"LogisticRegression", "RandomForest", "XGBoost", "LightGBM"}
    assert set(report.leaderboard["Model"].tolist()) == expected_models

    # 2. Verify champion model selected
    assert report.champion_model_name in expected_models
    assert report.champion_estimator is not None

    # 3. Verify metrics computed
    assert "CV ROC-AUC (Mean)" in report.leaderboard.columns
    assert report.leaderboard["CV ROC-AUC (Mean)"].iloc[0] >= 0.5

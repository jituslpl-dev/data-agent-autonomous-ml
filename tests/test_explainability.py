"""
Unit tests for Granular SHAP Explainability.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_agent.explainability import ExplainabilityEngine

def test_shap_explainability():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.uniform(20, 60, size=50),
        "income": np.random.uniform(30000, 90000, size=50),
        "score": np.random.uniform(500, 800, size=50),
    })
    y = (X["age"] * 0.05 + X["income"] * 0.0001 > 5).astype(int)

    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)

    explainer = ExplainabilityEngine()
    report = explainer.explain(rf, "RandomForest", X, sample_indices=[0, 1])

    assert len(report.global_importance) == 3
    assert len(report.top_features) == 3
    assert len(report.sample_explanations) == 2
    assert len(report.insights) > 0

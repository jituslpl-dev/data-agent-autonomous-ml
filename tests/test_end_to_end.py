"""
End-to-End Pipeline Integration Test.
"""

import os
import tempfile
import pandas as pd
import numpy as np
from data_agent.pipeline import DataAgent

def test_full_pipeline_end_to_end():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "age": np.random.uniform(18, 65, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
        "education": np.random.choice(["HighSchool", "Bachelors", "Masters"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    # Inject missing values
    df.loc[5:8, "income"] = np.nan

    agent = DataAgent(cv_splits=3, search_iterations=2)
    agent.fit(df, target_col="target")

    assert agent.is_fitted
    assert agent.tournament_report.champion_estimator is not None

    # Test predictions on new data
    df_test = df.drop(columns=["target"]).head(10)
    preds = agent.predict(df_test)
    assert len(preds) == 10

    probs = agent.predict_proba(df_test)
    assert probs.shape[0] == 10
    assert probs.shape[1] == 2

    # Test diagnostic report generation
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        report_path = f.name

    try:
        md_report = agent.generate_report(html_output_path=report_path)
        assert len(md_report) > 100
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 500
    finally:
        if os.path.exists(report_path):
            os.remove(report_path)

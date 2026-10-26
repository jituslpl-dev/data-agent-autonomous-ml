"""
Unit tests for Schema Inspection & Profiling.
"""

import pandas as pd
import numpy as np
from data_agent.schema import SchemaInspector

def test_schema_inspection():
    df = pd.DataFrame({
        "id": range(100),
        "constant": [42] * 100,
        "age": np.random.uniform(20, 70, size=100),
        "category": np.random.choice(["A", "B", "C"], size=100),
        "target": np.random.choice([0, 1], size=100),
    })

    inspector = SchemaInspector()
    report = inspector.inspect(df, target_col="target")

    assert report.total_rows == 100
    assert report.total_columns == 5
    assert report.task_type == "binary_classification"
    assert "age" in report.numeric_features
    assert "category" in report.categorical_features
    assert "id" in report.dropped_features
    assert "constant" in report.dropped_features

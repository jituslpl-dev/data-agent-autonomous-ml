"""
Unit tests for Automated Data Cleaning & Imputation.
"""

import pandas as pd
import numpy as np
from data_agent.schema import SchemaInspector
from data_agent.cleaning import DataCleaner

def test_data_cleaner():
    df = pd.DataFrame({
        "id": range(50),
        "income": [100.0, np.nan, 300.0, 50000.0] + [200.0] * 46, # Contains missing and outlier
        "city": ["New York", np.nan, "London", "Tokyo"] + ["London"] * 46,
        "target": [1, 0, 1, 0] + [1] * 46,
    })

    inspector = SchemaInspector()
    schema = inspector.inspect(df, target_col="target")

    cleaner = DataCleaner(handle_outliers=True)
    df_clean = cleaner.fit_transform(df.drop(columns=["target"]), schema)

    # 1. Verify missing values are fully imputed
    assert df_clean["income"].isna().sum() == 0
    assert df_clean["city"].isna().sum() == 0

    # 2. Verify outlier winsorization
    assert df_clean["income"].max() < 50000.0

    # 3. Verify ID column dropped
    assert "id" not in df_clean.columns

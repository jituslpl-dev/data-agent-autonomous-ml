"""
Unit tests for Automated Feature Engineering.
"""

import pandas as pd
import numpy as np
from data_agent.schema import SchemaInspector
from data_agent.feature_engineering import FeatureEngineer

def test_feature_engineering():
    df = pd.DataFrame({
        "num1": np.linspace(1, 100, 50),
        "num2": np.linspace(10, 500, 50),
        "cat_low": np.random.choice(["Red", "Green", "Blue"], size=50),
        "target": np.random.choice([0, 1], size=50),
    })

    inspector = SchemaInspector()
    schema = inspector.inspect(df, target_col="target")

    fe = FeatureEngineer(create_interactions=True)
    df_feats = fe.fit_transform(df.drop(columns=["target"]), schema)

    # Verify interaction feature generated
    assert any("num1_x_num2" in c or "num1_div_num2" in c for c in df_feats.columns)
    # Verify one-hot encoding columns generated
    assert any("cat_low" in c for c in df_feats.columns)
    # Verify all outputs are numeric
    assert df_feats.select_dtypes(include=[np.number]).shape[1] == df_feats.shape[1]

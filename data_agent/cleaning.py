"""
Automated Data Cleaning & Imputation Module.
Handles missing values, outlier treatment (Winsorization), and feature filtering without data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .schema import SchemaReport

class DataCleaner:
    """
    Automated data cleaning and imputation transformer.
    Learns statistical parameters (medians, modes, bounds) strictly during `fit`
    to guarantee zero data leakage into validation/test splits.
    """
    def __init__(
        self,
        handle_outliers: bool = True,
        outlier_method: str = "iqr", # 'iqr' or 'percentile'
        missing_threshold: float = 0.80, # Drop features with >80% missing
    ):
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.missing_threshold = missing_threshold

        # Fitted parameters
        self.impute_values: Dict[str, Any] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.dropped_columns: List[str] = []
        self.missing_indicators: List[str] = []

    def fit(self, df: pd.DataFrame, schema: SchemaReport) -> "DataCleaner":
        self.dropped_columns = list(schema.dropped_features)

        # Check for columns with excessive missingness
        for col, prof in schema.columns.items():
            if prof.missing_ratio > self.missing_threshold and col not in self.dropped_columns:
                self.dropped_columns.append(col)

        # Learn imputation values
        for col in schema.numeric_features:
            if col in self.dropped_columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                self.impute_values[col] = 0.0
                continue

            prof = schema.columns[col]
            # Use median for skewed features, mean for symmetric features
            if prof.skewness is not None and abs(prof.skewness) > 1.0:
                self.impute_values[col] = float(series.median())
            else:
                self.impute_values[col] = float(series.mean())

            # Track missingness indicator if >5% missing
            if prof.missing_ratio > 0.05:
                self.missing_indicators.append(col)

            # Compute outlier bounds
            if self.handle_outliers:
                if self.outlier_method == "iqr":
                    q25 = float(series.quantile(0.25))
                    q75 = float(series.quantile(0.75))
                    iqr = q75 - q25
                    lower = q25 - 2.5 * iqr
                    upper = q75 + 2.5 * iqr
                else: # percentile
                    lower = float(series.quantile(0.01))
                    upper = float(series.quantile(0.99))
                self.outlier_bounds[col] = (lower, upper)

        # Categorical imputations
        for col in schema.categorical_features:
            if col in self.dropped_columns:
                continue
            mode_vals = df[col].dropna().mode()
            self.impute_values[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "missing"

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()

        # 1. Drop filtered columns
        cols_to_drop = [c for c in self.dropped_columns if c in out_df.columns]
        if cols_to_drop:
            out_df = out_df.drop(columns=cols_to_drop)

        # 2. Add missing indicators
        for col in self.missing_indicators:
            if col in out_df.columns:
                out_df[f"{col}_was_missing"] = out_df[col].isna().astype(float)

        # 3. Impute missing values
        for col, val in self.impute_values.items():
            if col in out_df.columns:
                out_df[col] = out_df[col].fillna(val)

        # 4. Outlier capping
        if self.handle_outliers:
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in out_df.columns:
                    out_df[col] = out_df[col].clip(lower=lower, upper=upper)

        return out_df

    def fit_transform(self, df: pd.DataFrame, schema: SchemaReport) -> pd.DataFrame:
        return self.fit(df, schema).transform(df)

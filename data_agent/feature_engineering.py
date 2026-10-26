"""
Automated Feature Engineering Module.
Creates interaction features, group aggregations, log-transforms, encodings, and feature selection.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .schema import SchemaReport

class FeatureEngineer:
    """
    Automated Feature Engineering Engine.
    Engineers domain-agnostic numerical interactions, datetime expansions,
    frequency/one-hot encodings, and variance-based feature pruning.
    """
    def __init__(
        self,
        create_interactions: bool = True,
        max_interactions: int = 5,
        apply_log_transform: bool = True,
        variance_threshold: float = 1e-4,
    ):
        self.create_interactions = create_interactions
        self.max_interactions = max_interactions
        self.apply_log_transform = apply_log_transform
        self.variance_threshold = variance_threshold

        # Learned transformers
        self.log_features: List[str] = []
        self.frequency_maps: Dict[str, Dict[Any, float]] = {}
        self.one_hot_columns: Dict[str, List[str]] = {}
        self.interaction_pairs: List[Tuple[str, str, str]] = [] # (colA, colB, op: 'mul' or 'ratio')
        self.selected_features: List[str] = []

    def fit(self, df: pd.DataFrame, schema: SchemaReport) -> "FeatureEngineer":
        # 1. Identify highly skewed features for log1p transform
        if self.apply_log_transform:
            for col in schema.numeric_features:
                if col in df.columns:
                    series = df[col]
                    if (series >= 0).all():
                        skew = float(series.skew()) if series.count() > 3 else 0.0
                        if skew > 2.0:
                            self.log_features.append(col)

        # 2. Learn Frequency Encodings for high-cardinality categoricals
        for col in schema.categorical_features:
            if col in df.columns:
                prof = schema.columns.get(col)
                if prof and prof.inferred_type == "categorical_high":
                    counts = df[col].value_counts(normalize=True).to_dict()
                    self.frequency_maps[col] = counts
                else: # Low-cardinality one-hot encoding categories
                    cats = [str(c) for c in df[col].dropna().unique().tolist()]
                    self.one_hot_columns[col] = sorted(cats)

        # 3. Identify top interaction pairs among continuous numeric features
        if self.create_interactions and len(schema.numeric_features) >= 2:
            num_cols = [c for c in schema.numeric_features if c in df.columns]
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    if len(self.interaction_pairs) >= self.max_interactions:
                        break
                    colA, colB = num_cols[i], num_cols[j]
                    self.interaction_pairs.append((colA, colB, "mul"))
                    # If colB > 0 everywhere, can also do ratio
                    if (df[colB] > 0).all():
                        self.interaction_pairs.append((colA, colB, "ratio"))

        # Transform training data to learn final selected features
        transformed = self._engineer_features(df, schema)
        # Drop constant/zero-variance features
        variances = transformed.var(numeric_only=True)
        self.selected_features = variances[variances > self.variance_threshold].index.tolist()

        return self

    def _engineer_features(self, df: pd.DataFrame, schema: SchemaReport) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # 1. Base Numerics
        for col in schema.numeric_features:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 2. Log Transforms
        for col in self.log_features:
            if col in df.columns:
                out[f"{col}_log1p"] = np.log1p(np.maximum(df[col].values, 0))

        # 3. Datetime Expansion
        for col in schema.datetime_features:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                out[f"{col}_year"] = dt.dt.year.fillna(2000).astype(int)
                out[f"{col}_month"] = dt.dt.month.fillna(1).astype(int)
                out[f"{col}_day"] = dt.dt.day.fillna(1).astype(int)
                out[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
                out[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

        # 4. Frequency Encodings
        for col, fmap in self.frequency_maps.items():
            if col in df.columns:
                out[f"{col}_freq"] = df[col].map(fmap).fillna(0.0)

        # 5. One-Hot Encodings
        for col, cats in self.one_hot_columns.items():
            if col in df.columns:
                str_s = df[col].astype(str)
                for cat in cats:
                    out[f"{col}_{cat}"] = (str_s == cat).astype(float)

        # 6. Interaction Features
        for colA, colB, op in self.interaction_pairs:
            if colA in df.columns and colB in df.columns:
                if op == "mul":
                    out[f"{colA}_x_{colB}"] = df[colA] * df[colB]
                elif op == "ratio":
                    out[f"{colA}_div_{colB}"] = df[colA] / (df[colB] + 1e-6)

        return out

    def transform(self, df: pd.DataFrame, schema: SchemaReport) -> pd.DataFrame:
        transformed = self._engineer_features(df, schema)
        # Retain only selected features learned during fit
        available = [c for c in self.selected_features if c in transformed.columns]
        return transformed[available]

    def fit_transform(self, df: pd.DataFrame, schema: SchemaReport) -> pd.DataFrame:
        self.fit(df, schema)
        return self.transform(df, schema)

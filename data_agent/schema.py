"""
Schema Inspection and Semantic Data Profiling Module.
Automatically infers data types, detects anomalies, and determines the task type.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str # 'numeric_continuous', 'numeric_discrete', 'categorical_low', 'categorical_high', 'datetime', 'text', 'id'
    num_missing: int
    missing_ratio: float
    num_unique: int
    is_constant: bool
    is_id_like: bool
    skewness: Optional[float] = None
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class SchemaReport:
    total_rows: int
    total_columns: int
    target_column: Optional[str]
    task_type: str # 'binary_classification', 'multiclass_classification', 'regression'
    columns: Dict[str, ColumnProfile]
    numeric_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    dropped_features: List[str] # constant or id-like features


class SchemaInspector:
    """
    Autonomous Schema Inspector that scans raw tabular datasets,
    determines feature semantics, identifies anomalies, and infers machine learning task type.
    """
    def __init__(self, high_cardinality_threshold: int = 50, id_threshold: float = 0.95):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.id_threshold = id_threshold

    def inspect(self, df: pd.DataFrame, target_col: Optional[str] = None) -> SchemaReport:
        total_rows, total_cols = df.shape
        profiles: Dict[str, ColumnProfile] = {}
        numeric_feats: List[str] = []
        categorical_feats: List[str] = []
        datetime_feats: List[str] = []
        dropped_feats: List[str] = []

        # Determine task type if target_col specified
        task_type = "binary_classification"
        if target_col is not None and target_col in df.columns:
            target_series = df[target_col].dropna()
            num_unique_target = target_series.nunique()
            if np.issubdtype(target_series.dtype, np.number) and num_unique_target > 10:
                task_type = "regression"
            elif num_unique_target == 2:
                task_type = "binary_classification"
            else:
                task_type = "multiclass_classification"

        for col in df.columns:
            if col == target_col:
                continue

            series = df[col]
            num_missing = int(series.isna().sum())
            missing_ratio = float(num_missing / total_rows)
            num_unique = int(series.nunique())
            is_constant = bool(num_unique <= 1)
            is_id_name = "id" in col.lower() or col.lower().endswith("_id")
            is_id_values = (num_unique / max(1, total_rows) >= self.id_threshold and total_rows > 30 and (pd.api.types.is_integer_dtype(series) or series.dtype == "object"))
            is_id_like = bool(is_id_name and (is_id_values or num_unique == total_rows))

            # Detect Datetime
            inferred_type = "categorical_low"
            if pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "datetime"
                datetime_feats.append(col)
            elif is_constant or is_id_like:
                inferred_type = "id" if is_id_like else "constant"
                dropped_feats.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                if num_unique <= 5:
                    inferred_type = "numeric_discrete"
                    numeric_feats.append(col)
                else:
                    inferred_type = "numeric_continuous"
                    numeric_feats.append(col)
            else:
                # String / Categorical / Object
                # Check if parseable as datetime
                try_dt = False
                if series.dtype == "object":
                    sample = series.dropna().head(10).astype(str)
                    if sample.str.contains(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").all():
                        try_dt = True

                if try_dt:
                    inferred_type = "datetime"
                    datetime_feats.append(col)
                elif num_unique > self.high_cardinality_threshold:
                    inferred_type = "categorical_high"
                    categorical_feats.append(col)
                else:
                    inferred_type = "categorical_low"
                    categorical_feats.append(col)

            skew = float(series.skew()) if inferred_type in ("numeric_continuous", "numeric_discrete") and series.count() > 3 else None

            sample_vals = series.dropna().head(3).tolist()

            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                num_missing=num_missing,
                missing_ratio=missing_ratio,
                num_unique=num_unique,
                is_constant=is_constant,
                is_id_like=is_id_like,
                skewness=skew,
                sample_values=sample_vals,
            )

        return SchemaReport(
            total_rows=total_rows,
            total_columns=total_cols,
            target_column=target_col,
            task_type=task_type,
            columns=profiles,
            numeric_features=numeric_feats,
            categorical_features=categorical_feats,
            datetime_features=datetime_feats,
            dropped_features=dropped_feats,
        )

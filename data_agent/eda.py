"""
Exploratory Data Analysis (EDA) and Statistical Diagnostic Module.
Computes correlations, multicollinearity flags, target associations, and data quality metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

@dataclass
class EDAReport:
    num_rows: int
    num_columns: int
    duplicate_rows: int
    total_missing_cells: int
    memory_usage_mb: float
    target_summary: Dict[str, Any]
    high_correlation_pairs: List[Tuple[str, str, float]]
    feature_target_correlations: Dict[str, float]
    summary_stats: Dict[str, Dict[str, float]]


class ExploratoryAnalysis:
    """Autonomous Exploratory Data Analysis Engine."""
    def __init__(self, collinearity_threshold: float = 0.85):
        self.collinearity_threshold = collinearity_threshold

    def analyze(self, df: pd.DataFrame, target_col: Optional[str] = None) -> EDAReport:
        num_rows, num_cols = df.shape
        dup_rows = int(df.duplicated().sum())
        total_missing = int(df.isna().sum().sum())
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

        # Target summary
        target_info: Dict[str, Any] = {}
        target_corrs: Dict[str, float] = {}

        if target_col is not None and target_col in df.columns:
            target_series = df[target_col].dropna()
            target_info["num_unique"] = int(target_series.nunique())
            target_info["missing"] = int(df[target_col].isna().sum())
            if np.issubdtype(target_series.dtype, np.number):
                target_info["mean"] = float(target_series.mean())
                target_info["std"] = float(target_series.std())
                target_info["min"] = float(target_series.min())
                target_info["max"] = float(target_series.max())
            else:
                target_info["class_distribution"] = {
                    str(k): int(v) for k, v in target_series.value_counts().items()
                }

        # Numeric correlations and collinearity
        numeric_df = df.select_dtypes(include=[np.number])
        collinear_pairs: List[Tuple[str, str, float]] = []

        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

            for col_i, col1 in enumerate(corr_matrix.columns):
                for col_j, col2 in enumerate(corr_matrix.columns):
                    if upper_tri[col_i, col_j]:
                        r_val = float(corr_matrix.iloc[col_i, col_j])
                        if r_val >= self.collinearity_threshold:
                            collinear_pairs.append((col1, col2, round(r_val, 3)))

            # Correlations with target
            if target_col is not None and target_col in numeric_df.columns:
                target_corr_series = numeric_df.corr()[target_col].drop(labels=[target_col]).dropna()
                target_corrs = {
                    col: round(float(val), 4)
                    for col, val in target_corr_series.sort_values(ascending=False).items()
                }

        # Summary statistics per numeric column
        summary_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if len(s) > 0:
                summary_stats[col] = {
                    "mean": round(float(s.mean()), 3),
                    "std": round(float(s.std()), 3),
                    "median": round(float(s.median()), 3),
                    "min": round(float(s.min()), 3),
                    "max": round(float(s.max()), 3),
                }

        return EDAReport(
            num_rows=num_rows,
            num_columns=num_cols,
            duplicate_rows=dup_rows,
            total_missing_cells=total_missing,
            memory_usage_mb=round(mem_mb, 2),
            target_summary=target_info,
            high_correlation_pairs=collinear_pairs,
            feature_target_correlations=target_corrs,
            summary_stats=summary_stats,
        )

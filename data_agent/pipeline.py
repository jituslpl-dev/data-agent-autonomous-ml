"""
Main DataAgent Autonomous Machine Learning Pipeline Orchestrator.
Integrates schema inspection, data cleaning, EDA, feature engineering, 4-model stratified CV, and SHAP explainability.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
from .schema import SchemaInspector, SchemaReport
from .cleaning import DataCleaner
from .eda import ExploratoryAnalysis, EDAReport
from .feature_engineering import FeatureEngineer
from .models.trainer import ModelTournamentTrainer, TournamentReport
from .explainability import ExplainabilityEngine, ExplainabilityReport
from .report import DiagnosticReportGenerator

class DataAgent:
    """
    End-to-End Autonomous Data Science & Machine Learning Agent.
    Transforms raw tabular datasets into optimized, cross-validated predictive models with SHAP explanations.
    """
    def __init__(
        self,
        cv_splits: int = 5,
        search_iterations: int = 4,
        handle_outliers: bool = True,
        create_interactions: bool = True,
        random_state: int = 42,
    ):
        self.cv_splits = cv_splits
        self.search_iterations = search_iterations
        self.handle_outliers = handle_outliers
        self.create_interactions = create_interactions
        self.random_state = random_state

        # Pipeline components
        self.schema_inspector = SchemaInspector()
        self.eda_engine = ExploratoryAnalysis()
        self.cleaner = DataCleaner(handle_outliers=handle_outliers)
        self.feature_engineer = FeatureEngineer(create_interactions=create_interactions)
        self.trainer = ModelTournamentTrainer(
            n_splits=cv_splits,
            search_iterations=search_iterations,
            random_state=random_state,
        )
        self.explainability_engine = ExplainabilityEngine()

        # State reports
        self.schema_report: Optional[SchemaReport] = None
        self.eda_report: Optional[EDAReport] = None
        self.tournament_report: Optional[TournamentReport] = None
        self.shap_report: Optional[ExplainabilityReport] = None
        self.target_col: Optional[str] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, target_col: str) -> "DataAgent":
        """
        Runs the full autonomous machine learning pipeline:
        1. Raw Schema Inspection & Type Classification
        2. Exploratory Data Analysis & Multicollinearity Profiling
        3. Automated Cleaning & Outlier Winsorization
        4. Automated Feature Engineering & Selection
        5. Stratified CV Tournament across 4 Model Families
        6. Granular SHAP Attribution
        """
        self.target_col = target_col
        y = df[target_col]
        X_raw = df.drop(columns=[target_col])

        # 1. Schema Inspection
        self.schema_report = self.schema_inspector.inspect(df, target_col=target_col)

        # 2. Exploratory Data Analysis
        self.eda_report = self.eda_engine.analyze(df, target_col=target_col)

        # 3. Data Cleaning
        X_cleaned = self.cleaner.fit_transform(X_raw, self.schema_report)

        # 4. Feature Engineering
        X_features = self.feature_engineer.fit_transform(X_cleaned, self.schema_report)

        # 5. Model Tournament Training (Stratified Cross-Validation across 4 model families)
        self.tournament_report = self.trainer.train_and_evaluate(
            X_features,
            y,
            task_type=self.schema_report.task_type,
        )

        # 6. Granular SHAP Explainability
        self.shap_report = self.explainability_engine.explain(
            model=self.tournament_report.champion_estimator,
            model_name=self.tournament_report.champion_model_name,
            X=X_features,
        )

        self.is_fitted = True
        return self

    def _prepare_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("DataAgent must be fitted before running predict().")
        df_eval = df.drop(columns=[self.target_col]) if self.target_col in df.columns else df
        df_cleaned = self.cleaner.transform(df_eval)
        df_feats = self.feature_engineer.transform(df_cleaned, self.schema_report)
        return df_feats

    def predict(self, df: pd.DataFrame):
        """Generates predictions using the champion model."""
        X_eval = self._prepare_inference_features(df)
        return self.tournament_report.champion_estimator.predict(X_eval)

    def predict_proba(self, df: pd.DataFrame):
        """Generates class prediction probabilities using the champion model."""
        X_eval = self._prepare_inference_features(df)
        if hasattr(self.tournament_report.champion_estimator, "predict_proba"):
            return self.tournament_report.champion_estimator.predict_proba(X_eval)
        raise NotImplementedError(f"Model {self.tournament_report.champion_model_name} does not support predict_proba.")

    def generate_report(self, html_output_path: Optional[str] = None) -> str:
        """Generates markdown summary and optional self-contained HTML diagnostic dashboard."""
        if not self.is_fitted:
            raise RuntimeError("DataAgent must be fitted before generating reports.")

        if html_output_path is not None:
            DiagnosticReportGenerator.generate_html(
                schema=self.schema_report,
                cleaner=self.cleaner,
                eda=self.eda_report,
                tournament=self.tournament_report,
                shap_report=self.shap_report,
                output_filepath=html_output_path,
            )

        return DiagnosticReportGenerator.generate_markdown(
            schema=self.schema_report,
            cleaner=self.cleaner,
            eda=self.eda_report,
            tournament=self.tournament_report,
            shap_report=self.shap_report,
        )

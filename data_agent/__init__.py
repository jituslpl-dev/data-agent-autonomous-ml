"""
DataAgent: Autonomous Data Science & Machine Learning Agent.
"""

from .pipeline import DataAgent
from .schema import SchemaInspector, SchemaReport
from .cleaning import DataCleaner
from .eda import ExploratoryAnalysis, EDAReport
from .feature_engineering import FeatureEngineer
from .models.trainer import ModelTournamentTrainer, TournamentReport
from .explainability import ExplainabilityEngine, ExplainabilityReport
from .report import DiagnosticReportGenerator

__all__ = [
    "DataAgent",
    "SchemaInspector",
    "SchemaReport",
    "DataCleaner",
    "ExploratoryAnalysis",
    "EDAReport",
    "FeatureEngineer",
    "ModelTournamentTrainer",
    "TournamentReport",
    "ExplainabilityEngine",
    "ExplainabilityReport",
    "DiagnosticReportGenerator",
]

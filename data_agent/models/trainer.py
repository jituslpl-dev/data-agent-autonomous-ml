"""
Stratified Cross-Validation Tournament Trainer across 4 Model Families.
Automates hyperparameter search, multi-metric scoring, and champion model selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
)
from .registry import get_model_families

@dataclass
class ModelEvaluationResult:
    model_name: str
    best_params: Dict[str, Any]
    cv_primary_score_mean: float
    cv_primary_score_std: float
    metrics: Dict[str, float]
    trained_estimator: Any


@dataclass
class TournamentReport:
    task_type: str
    primary_metric: str
    leaderboard: pd.DataFrame
    champion_model_name: str
    champion_estimator: Any
    all_results: Dict[str, ModelEvaluationResult]


class ModelTournamentTrainer:
    """
    Automated ML Tournament Trainer.
    Runs Stratified Cross-Validation across Logistic Regression, Random Forest, XGBoost, and LightGBM.
    """
    def __init__(
        self,
        n_splits: int = 5,
        search_iterations: int = 4,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.search_iterations = search_iterations
        self.random_state = random_state

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "binary_classification",
    ) -> TournamentReport:
        is_classification = "classification" in task_type
        is_binary = task_type == "binary_classification"

        if is_classification:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scoring_metric = "roc_auc" if is_binary else "f1_weighted"
            primary_metric_name = "ROC-AUC" if is_binary else "F1-Score"
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scoring_metric = "r2"
            primary_metric_name = "R-Squared"

        model_families = get_model_families(task_type, random_state=self.random_state)
        results: Dict[str, ModelEvaluationResult] = {}
        leaderboard_rows: List[Dict[str, Any]] = []

        for name, spec in model_families.items():
            estimator = spec["estimator"]
            param_grid = spec["param_grid"]

            # Configurable hyperparameter search
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=min(self.search_iterations, len(param_grid)),
                scoring=scoring_metric,
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )

            search.fit(X, y)
            best_model = search.best_estimator_
            best_score = float(search.best_score_)
            std_score = float(search.cv_results_["std_test_score"][search.best_index_])

            # Calculate secondary evaluation metrics on full dataset
            y_pred = best_model.predict(X)
            metrics: Dict[str, float] = {}

            if is_classification:
                metrics["Accuracy"] = round(float(accuracy_score(y, y_pred)), 4)
                metrics["F1"] = round(float(f1_score(y, y_pred, average="weighted")), 4)
                metrics["Precision"] = round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4)
                metrics["Recall"] = round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4)
                if is_binary and hasattr(best_model, "predict_proba"):
                    try:
                        y_prob = best_model.predict_proba(X)[:, 1]
                        metrics["ROC-AUC"] = round(float(roc_auc_score(y, y_prob)), 4)
                    except Exception:
                        metrics["ROC-AUC"] = round(best_score, 4)
                else:
                    metrics["ROC-AUC"] = round(best_score, 4)
            else:
                metrics["R2"] = round(float(r2_score(y, y_pred)), 4)
                metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)

            eval_res = ModelEvaluationResult(
                model_name=name,
                best_params=search.best_params_,
                cv_primary_score_mean=round(best_score, 4),
                cv_primary_score_std=round(std_score, 4),
                metrics=metrics,
                trained_estimator=best_model,
            )
            results[name] = eval_res

            leaderboard_rows.append({
                "Model": name,
                f"CV {primary_metric_name} (Mean)": round(best_score, 4),
                "CV Std": round(std_score, 4),
                **metrics,
            })

        # Rank leaderboard by primary metric descending
        df_leaderboard = pd.DataFrame(leaderboard_rows)
        sort_col = f"CV {primary_metric_name} (Mean)"
        df_leaderboard = df_leaderboard.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        champion_name = str(df_leaderboard.iloc[0]["Model"])
        champion_estimator = results[champion_name].trained_estimator

        return TournamentReport(
            task_type=task_type,
            primary_metric=primary_metric_name,
            leaderboard=df_leaderboard,
            champion_model_name=champion_name,
            champion_estimator=champion_estimator,
            all_results=results,
        )

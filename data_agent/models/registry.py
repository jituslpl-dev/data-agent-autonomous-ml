"""
Model Registry defining 4 Model Families (Logistic Regression, Random Forest, XGBoost, LightGBM)
and their configurable hyperparameter search grids.
"""

from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

def get_model_families(task_type: str = "binary_classification", random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns the 4 model families configured with estimators and hyperparameter search grids.
    Supports binary classification, multiclass classification, and regression.
    """
    is_classification = "classification" in task_type

    if is_classification:
        models = {
            "LogisticRegression": {
                "estimator": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(random_state=random_state, max_iter=1000))
                ]),
                "param_grid": {
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                    "model__solver": ["lbfgs"],
                }
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                }
            },
            "XGBoost": {
                "estimator": xgb.XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0
                ),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                }
            },
            "LightGBM": {
                "estimator": lgb.LGBMClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                ),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "num_leaves": [15, 31],
                    "learning_rate": [0.05, 0.1],
                }
            }
        }
    else: # Regression
        models = {
            "LinearRegression": {
                "estimator": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=random_state))
                ]),
                "param_grid": {
                    "model__alpha": [0.1, 1.0, 10.0, 100.0],
                }
            },
            "RandomForest": {
                "estimator": RandomForestRegressor(random_state=random_state, n_jobs=-1),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                }
            },
            "XGBoost": {
                "estimator": xgb.XGBRegressor(
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0
                ),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                }
            },
            "LightGBM": {
                "estimator": lgb.LGBMRegressor(
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                ),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "num_leaves": [15, 31],
                    "learning_rate": [0.05, 0.1],
                }
            }
        }

    return models

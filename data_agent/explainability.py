"""
Granular SHAP Explainability Engine.
Computes global feature attributions, local waterfall sample contributions, and directional impact insights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap

@dataclass
class LocalExplanation:
    sample_index: int
    base_value: float
    predicted_value: float
    top_positive_features: List[Dict[str, Any]]
    top_negative_features: List[Dict[str, Any]]


@dataclass
class ExplainabilityReport:
    model_name: str
    global_importance: Dict[str, float]
    top_features: List[str]
    sample_explanations: List[LocalExplanation]
    insights: List[str]


class ExplainabilityEngine:
    """
    Granular SHAP Explainability Engine.
    Employs TreeExplainer for Tree Ensemble models and Linear/Exact Explainer for linear pipelines.
    """
    def __init__(self, max_background_samples: int = 100):
        self.max_background_samples = max_background_samples

    def explain(
        self,
        model: Any,
        model_name: str,
        X: pd.DataFrame,
        sample_indices: Optional[List[int]] = None,
    ) -> ExplainabilityReport:
        if sample_indices is None:
            sample_indices = [0, 1, 2] if len(X) >= 3 else list(range(len(X)))

        # Subsample background data if dataset is large
        if len(X) > self.max_background_samples:
            X_background = shap.sample(X, self.max_background_samples, random_state=42)
        else:
            X_background = X

        # Unwrap pipeline if model is a Pipeline
        unwrapped_model = model
        X_eval = X
        if hasattr(model, "named_steps"):
            # Transform X using preprocessing steps before final estimator
            if "scaler" in model.named_steps:
                X_eval = pd.DataFrame(
                    model.named_steps["scaler"].transform(X),
                    columns=X.columns,
                    index=X.index,
                )
            unwrapped_model = model.named_steps["model"]

        # Instantiate appropriate SHAP explainer
        try:
            if "Forest" in model_name or "XGB" in model_name or "LGB" in model_name:
                explainer = shap.TreeExplainer(unwrapped_model)
                shap_values = explainer.shap_values(X_eval)
            else:
                explainer = shap.LinearExplainer(unwrapped_model, X_eval)
                shap_values = explainer.shap_values(X_eval)
        except Exception:
            # Fallback to Exact/Kernel explainer
            explainer = shap.Explainer(unwrapped_model.predict, X_eval.iloc[:20])
            shap_values = explainer(X_eval).values

        # Normalize shap_values array shape
        if isinstance(shap_values, list):
            # Binary classification (take positive class shap values)
            vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = shap_values

        if vals.ndim == 3:
            vals = vals[:, :, 1] # Take positive class if 3D

        # 1. Global Feature Importance (Mean Absolute SHAP)
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        feature_names = list(X.columns)

        importance_dict = {
            feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(feature_names))
        }
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        top_features = list(sorted_importance.keys())[:10]

        # 2. Local Sample Attributions
        local_explanations: List[LocalExplanation] = []
        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") and np.isscalar(explainer.expected_value) else 0.0

        for idx in sample_indices:
            if idx >= len(X):
                continue
            sample_shap = vals[idx]
            sample_features = X.iloc[idx].to_dict()

            pos_contribs = []
            neg_contribs = []

            for i, f in enumerate(feature_names):
                impact = float(sample_shap[i])
                val = sample_features[f]
                item = {"feature": f, "value": val, "impact": round(impact, 4)}
                if impact > 0:
                    pos_contribs.append(item)
                else:
                    neg_contribs.append(item)

            pos_contribs.sort(key=lambda x: x["impact"], reverse=True)
            neg_contribs.sort(key=lambda x: x["impact"])

            pred_val = float(model.predict(X.iloc[[idx]])[0])

            local_explanations.append(
                LocalExplanation(
                    sample_index=idx,
                    base_value=round(base_val, 4),
                    predicted_value=round(pred_val, 4),
                    top_positive_features=pos_contribs[:5],
                    top_negative_features=neg_contribs[:5],
                )
            )

        # 3. Generate Natural Language Insights
        insights = []
        if top_features:
            insights.append(f"Primary driving feature is '{top_features[0]}' with mean SHAP impact of {sorted_importance[top_features[0]]:.4f}.")
        if len(top_features) > 1:
            insights.append(f"Top 3 influential features across dataset: {', '.join(top_features[:3])}.")

        return ExplainabilityReport(
            model_name=model_name,
            global_importance=sorted_importance,
            top_features=top_features,
            sample_explanations=local_explanations,
            insights=insights,
        )

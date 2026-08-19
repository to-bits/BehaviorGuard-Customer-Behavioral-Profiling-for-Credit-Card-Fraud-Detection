"""SHAP explanations for the existing BehaviorGuard XGBoost pipeline.

The explanations describe model contribution, not causal influence.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap


class BehaviorGuardExplainer:
    """Explain the fitted XGBoost model without changing or retraining it."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        try:
            self.preprocessor = pipeline.named_steps["preprocessor"]
            self.model = pipeline.named_steps["model"]
        except (AttributeError, KeyError) as exc:
            raise TypeError(
                "pipeline must contain fitted 'preprocessor' and 'model' steps"
            ) from exc

        self._tree_explainer = shap.TreeExplainer(self.model)
        self._feature_groups = self._build_feature_groups()

    def _build_feature_groups(self) -> dict[str, list[int]]:
        """Map transformed columns back to their original input features."""
        groups: dict[str, list[int]] = {}
        output_indices = self.preprocessor.output_indices_

        for transformer_name, transformer, columns in self.preprocessor.transformers_:
            if transformer_name == "remainder" or transformer_name not in output_indices:
                continue

            output_slice = output_indices[transformer_name]
            transformed_names = self.preprocessor.named_transformers_[
                transformer_name
            ].get_feature_names_out(columns)
            transformed_names = list(transformed_names)
            start = output_slice.start or 0

            if transformer_name == "cat":
                for feature in columns:
                    prefix = f"{feature}_"
                    matching = [
                        start + index
                        for index, name in enumerate(transformed_names)
                        if name.startswith(prefix)
                    ]
                    if matching:
                        groups[feature] = matching
            else:
                for index, feature in enumerate(columns):
                    groups[feature] = [start + index]

        return groups

    def _validate_features(self, features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty:
            raise ValueError("features must contain at least one row")

        expected = list(self.pipeline.feature_names_in_)
        missing = [feature for feature in expected if feature not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

    def _shap_values(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._validate_features(features)
        model_features = features.loc[:, list(self.pipeline.feature_names_in_)]
        transformed = self.preprocessor.transform(model_features)
        values = np.asarray(self._tree_explainer.shap_values(transformed))

        if values.ndim == 3:
            values = values[:, :, 1]
        if values.ndim != 2:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

        base_values = np.asarray(self._tree_explainer.expected_value)
        if base_values.ndim == 0:
            base_values = np.repeat(float(base_values), len(features))
        else:
            base_values = np.repeat(float(base_values[-1]), len(features))

        return values, base_values

    def global_feature_importance(
        self,
        features: pd.DataFrame,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return mean absolute SHAP contribution by original feature."""
        values, _ = self._shap_values(features)
        rows = []

        for feature, indices in self._feature_groups.items():
            contribution = np.abs(values[:, indices]).sum(axis=1).mean()
            rows.append(
                {
                    "feature": feature,
                    "mean_abs_shap": float(contribution),
                }
            )

        rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
        return rows if top_n is None else rows[:top_n]

    def local_contributions(
        self,
        features: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Return signed local SHAP contributions for one transaction."""
        self._validate_features(features)
        if len(features) != 1:
            raise ValueError("local_contributions expects exactly one transaction row")

        values, base_values = self._shap_values(features)
        row = features.iloc[0]
        contributions = []

        for feature, indices in self._feature_groups.items():
            contribution = float(values[0, indices].sum())
            numeric_value = row[feature]
            display_value = _display_value(numeric_value)
            contributions.append(
                {
                    "feature": feature,
                    "contribution": contribution,
                    "direction": "toward_fraud" if contribution >= 0 else "reduces_risk",
                    "display_value": display_value,
                }
            )

        contributions.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )
        return contributions

    def explain_prediction(
        self,
        features: pd.DataFrame,
        prediction: dict[str, Any],
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Combine an independent prediction result with local SHAP factors."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        local = self.local_contributions(features)
        positive = [item for item in local if item["contribution"] > 0][:top_n]
        negative = [item for item in local if item["contribution"] < 0][:top_n]

        return {
            **prediction,
            "top_factors": local[:top_n],
            "top_positive_contributors": positive,
            "top_negative_contributors": negative,
            "shap_base_value": float(self._shap_values(features)[1][0]),
            "explanation_note": (
                "Contributions describe how the model influenced this prediction; "
                "they do not prove causality."
            ),
        }


def _display_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def explain_prediction(
    pipeline: Any,
    features: pd.DataFrame,
    prediction: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    """Convenience wrapper for a single machine-readable explanation."""
    return BehaviorGuardExplainer(pipeline).explain_prediction(
        features=features,
        prediction=prediction,
        top_n=top_n,
    )
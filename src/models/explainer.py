"""
SHAP explanations: which features pushed each prediction's risk up or down.
Used by the API, the dashboard and the RAG index.
"""

import numpy as np
import pandas as pd
import shap

from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def build_explainer(model, background: np.ndarray | None = None):
    """
    Pick a SHAP explainer by model type: TreeExplainer for tree models,
    LinearExplainer for logistic regression, KernelExplainer otherwise.
    """
    try:
        name = type(model).__name__

        if name in {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
                    "GradientBoostingClassifier", "DecisionTreeClassifier"}:
            logger.info(f"TreeExplainer built for {name}")
            return shap.TreeExplainer(model)

        if name in {"LogisticRegression", "LinearRegression", "RidgeClassifier", "SGDClassifier"}:
            if background is None:
                raise ValueError("LinearExplainer requires a background sample")
            logger.info(f"LinearExplainer built for {name}")
            return shap.LinearExplainer(model, background)

        if background is None:
            raise ValueError(f"KernelExplainer requires a background sample for {name}")
        logger.info(f"KernelExplainer (model-agnostic fallback) built for {name}")
        return shap.KernelExplainer(model.predict_proba, shap.sample(background, 100))

    except Exception as e:
        raise SentrixException(e, sys)


# Alias for older call sites.
def build_tree_explainer(model, background: np.ndarray | None = None):
    return build_explainer(model, background)


def explain_predictions(explainer, X: np.ndarray) -> np.ndarray:
    """
    SHAP values as an (n_samples, n_features) array for the positive class.

    SHAP returns different shapes depending on explainer and version (a list
    per class, a 3-D array, or a 2-D array), so all three are handled.
    """
    try:
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) — keep the positive class
            shap_values = shap_values[..., -1]

        if shap_values.ndim != 2:
            raise ValueError(
                f"Expected 2-D SHAP values (n_samples, n_features), got shape "
                f"{shap_values.shape}. The explainer returned a layout this "
                f"function does not handle."
            )
        return shap_values
    except Exception as e:
        raise SentrixException(e, sys)


def explain_single_prediction(explainer, x_row: np.ndarray, feature_names: list[str],
                               top_n: int = 5) -> list[dict]:
    """Top_n features for one prediction as [{feature, contribution}] (positive = raises risk)."""
    try:
        shap_values = explain_predictions(explainer, x_row.reshape(1, -1))[0]

        if len(shap_values) != len(feature_names):
            raise ValueError(
                f"{len(shap_values)} SHAP values for {len(feature_names)} feature "
                f"names — the attribution would be mapped to the wrong columns."
            )

        contributions = list(zip(feature_names, (float(v) for v in shap_values)))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        return [
            {"feature": name, "contribution": round(float(value), 4)}
            for name, value in contributions[:top_n]
        ]
    except Exception as e:
        raise SentrixException(e, sys)


def get_readable_feature_names(numeric_cols: list[str], categorical_cols: list[str],
                                preprocessor) -> list[str]:
    """Feature names in the same column order as the one-hot encoded matrix."""
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
        return numeric_cols + cat_names
    except Exception as e:
        raise SentrixException(e, sys)


def global_feature_importance(explainer, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature across the test set."""
    try:
        shap_values = explain_predictions(explainer, X)
        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception as e:
        raise SentrixException(e, sys)

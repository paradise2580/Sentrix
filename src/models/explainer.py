"""
src/models/explainer.py

Role
----
For every prediction, SHAP tells us WHICH features drove the risk score
up or down. This turns a black-box probability into an actionable,
defensible explanation — and, critically, its output feeds directly into
the RAG chat layer, so a user asking "why is Supplier X risky?"
gets an answer grounded in the model's actual reasoning, not a guess.
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
    Pick the right SHAP explainer for the model type.

    TreeExplainer  — exact and fast for tree ensembles (RF / XGBoost / LightGBM).
    LinearExplainer — exact for linear models (Logistic Regression). Needed
                      because evaluation on the real Olist data selected
                      Logistic Regression as the best model, and TreeExplainer
                      cannot explain it.
    KernelExplainer — model-agnostic fallback (slow; used only if neither fits).

    Choosing by model type rather than assuming trees keeps explainability
    working no matter which model wins evaluation.
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


# Backwards-compatible alias — older call sites used the tree-specific name.
def build_tree_explainer(model, background: np.ndarray | None = None):
    return build_explainer(model, background)


def explain_predictions(explainer, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a batch of feature rows."""
    try:
        shap_values = explainer.shap_values(X)
        # Some explainers return a list per class; normalize to the positive-class array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values
    except Exception as e:
        raise SentrixException(e, sys)


def explain_single_prediction(explainer, x_row: np.ndarray, feature_names: list[str],
                               top_n: int = 5) -> list[dict]:
    """
    Explain ONE prediction in the plain-language shape the dashboard and
    RAG chat consume: a ranked list of {feature, contribution} for the
    top_n most influential features, sign included (positive = raises risk).
    """
    try:
        shap_values = explain_predictions(explainer, x_row.reshape(1, -1))[0]

        contributions = list(zip(feature_names, shap_values))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        return [
            {"feature": name, "contribution": round(float(value), 4)}
            for name, value in contributions[:top_n]
        ]
    except Exception as e:
        raise SentrixException(e, sys)


def get_readable_feature_names(numeric_cols: list[str], categorical_cols: list[str],
                                preprocessor) -> list[str]:
    """
    After OneHotEncoder, column order no longer matches the original
    DataFrame — this reconstructs human-readable names in the exact order
    the transformed matrix columns are in, so SHAP output maps back to
    real feature names instead of 'feature_17'.
    """
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
        return numeric_cols + cat_names
    except Exception as e:
        raise SentrixException(e, sys)


def global_feature_importance(explainer, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature across the whole test set — the global ranking."""
    try:
        shap_values = explain_predictions(explainer, X)
        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception as e:
        raise SentrixException(e, sys)

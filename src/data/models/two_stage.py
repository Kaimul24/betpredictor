from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


MARKET_PROBABILITY_COL = "p_open_home_median_nv"
PRETRAINED_BASEBALL_LOGIT_COL = "pretrained_baseball_logit"

MARKET_FEATURE_COLUMNS = {
    "vig_open",
    "num_books",
    "logit_prob_home_std_nv",
}
MARKET_FEATURE_PREFIXES = (
    "p_open_",
    "home_opening_prob_",
    "away_opening_prob_",
    "home_opening_logit_",
    "away_opening_logit_",
)

def is_market_feature_column(column: str) -> bool:
    return column in MARKET_FEATURE_COLUMNS or column.startswith(MARKET_FEATURE_PREFIXES)

def baseball_feature_columns(columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in columns
        if column != PRETRAINED_BASEBALL_LOGIT_COL and not is_market_feature_column(column)
    ]

def _flatten(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)

def _clip_probabilities(values: Any, eps: float = 1e-6) -> np.ndarray:
    probabilities = _flatten(values)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability predictions must be finite.")
    return np.clip(probabilities, eps, 1 - eps)

def get_market_baseline_predictions(
    model_data: dict[str, Any],
    split: str = "test",
    eps: float = 1e-6,
) -> np.ndarray:
    key = f"X_{split}"
    if key not in model_data:
        raise KeyError(f"model_data is missing {key}.")

    X = model_data[key]
    if MARKET_PROBABILITY_COL not in X.columns:
        raise ValueError(f"{MARKET_PROBABILITY_COL} is required for market baseline predictions.")

    return _clip_probabilities(X[MARKET_PROBABILITY_COL].to_numpy(dtype=float), eps=eps)

def evaluate_probability_predictions(y_true: Any, p_pred: Any) -> dict[str, float]:
    y = _flatten(y_true)
    p = _clip_probabilities(p_pred)
    if len(y) != len(p):
        raise ValueError(f"y_true and p_pred length mismatch: {len(y)} != {len(p)}.")

    roc_auc = np.nan
    if len(np.unique(y)) == 2:
        roc_auc = float(roc_auc_score(y, p))

    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": roc_auc,
    }

def evaluate_market_baseline(model_data: dict[str, Any], split: str = "test") -> dict[str, float]:
    y_key = f"y_{split}"
    if y_key not in model_data:
        raise KeyError(f"model_data is missing {y_key}.")

    return evaluate_probability_predictions(
        model_data[y_key],
        get_market_baseline_predictions(model_data, split=split),
    )

def validate_baseball_feature_signature(
    pretrained_columns: Sequence[str],
    finetune_columns: Sequence[str],
) -> None:
    expected = list(pretrained_columns)
    actual = baseball_feature_columns(finetune_columns)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "Pretrained baseball feature columns do not match finetune baseball feature columns. "
            f"Missing in finetune: {missing}. Extra in finetune: {extra}."
        )


def add_pretrained_logit_feature(
    model_data: dict[str, Any],
    feature_names: Sequence[str],
    predict_proba: Callable,
    logit: Callable,
    output_col: str = PRETRAINED_BASEBALL_LOGIT_COL,
) -> dict[str, Any]:
    if not feature_names:
        raise ValueError("Pretrained model has no feature names.")

    updated = model_data.copy()
    for key in ["X_train", "X_val", "X_test"]:
        X = model_data[key].copy()
        validate_baseball_feature_signature(feature_names, X.columns)
        X[output_col] = logit(predict_proba(X.loc[:, feature_names]))
        updated[key] = X

    return updated

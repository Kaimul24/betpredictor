from collections.abc import Callable, Sequence
from typing import Any


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

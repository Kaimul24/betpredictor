from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.models.two_stage import (
    MARKET_PROBABILITY_COL,
    PRETRAINED_BASEBALL_LOGIT_COL,
    add_pretrained_logit_feature,
    baseball_feature_columns,
    validate_baseball_feature_signature,
)
from src.data.models.xgboost_model import XGBoostModel, add_pretrained_xgboost_logits


def _model_args(training_mode, stage):
    return SimpleNamespace(training_mode=training_mode, stage=stage, retune=False)


def _model_data(include_market):
    index = pd.Index(range(8), name="row")
    X = pd.DataFrame(
        {
            "feature_a": np.linspace(0.1, 0.8, 8),
            "feature_b": np.linspace(1.0, 2.0, 8),
        },
        index=index,
    )
    if include_market:
        X[MARKET_PROBABILITY_COL] = np.linspace(0.45, 0.55, 8)

    y = pd.DataFrame({"is_winner_home": [0, 1, 0, 1, 1, 0, 1, 0]}, index=index)
    return {
        "X_train": X.iloc[:4].copy(),
        "y_train": y.iloc[:4].copy(),
        "X_val": X.iloc[4:6].copy(),
        "y_val": y.iloc[4:6].copy(),
        "X_test": X.iloc[6:].copy(),
        "y_test": y.iloc[6:].copy(),
    }


def test_xgboost_baseball_only_accepts_data_without_market_probability():
    model_data = _model_data(include_market=False)

    model = XGBoostModel(
        _model_args("baseball_only", "pretrain"),
        model_data,
    )

    assert model.uses_market_base_margin is False
    assert model.p_mkt_train is None
    assert MARKET_PROBABILITY_COL not in model.X_train.columns


def test_xgboost_market_residual_requires_market_probability():
    with pytest.raises(ValueError, match=MARKET_PROBABILITY_COL):
        XGBoostModel(
            _model_args("market_residual", "finetune"),
            _model_data(include_market=False),
        )


class FakePretrainedBooster:
    feature_names = ["feature_a", "feature_b"]

    def predict(self, dmatrix):
        return np.full(dmatrix.num_row(), 0.6)


def test_add_pretrained_xgboost_logits_keeps_market_column_and_adds_logit():
    model_data = _model_data(include_market=True)

    updated = add_pretrained_xgboost_logits(model_data, FakePretrainedBooster())

    assert MARKET_PROBABILITY_COL in updated["X_train"].columns
    assert PRETRAINED_BASEBALL_LOGIT_COL in updated["X_train"].columns
    assert updated["X_train"][PRETRAINED_BASEBALL_LOGIT_COL].iloc[0] == pytest.approx(
        XGBoostModel.logit(np.array([0.6]))[0]
    )
    assert PRETRAINED_BASEBALL_LOGIT_COL not in model_data["X_train"].columns


def test_add_pretrained_logit_feature_uses_callback_and_keeps_market_columns():
    model_data = _model_data(include_market=True)
    for X in [model_data["X_train"], model_data["X_val"], model_data["X_test"]]:
        X["vig_open"] = 1.05
        X["num_books"] = 2
        X["p_open_mean_nv_diff"] = 0.1
        X["logit_prob_home_std_nv"] = 0.0

    seen_columns = []

    def predict_proba(X):
        seen_columns.append(list(X.columns))
        return np.full(len(X), 0.7)

    updated = add_pretrained_logit_feature(
        model_data,
        ["feature_a", "feature_b"],
        predict_proba,
        XGBoostModel.logit,
    )

    assert seen_columns == [["feature_a", "feature_b"]] * 3
    assert MARKET_PROBABILITY_COL in updated["X_train"].columns
    assert "vig_open" in updated["X_train"].columns
    assert "p_open_mean_nv_diff" in updated["X_train"].columns
    assert updated["X_train"][PRETRAINED_BASEBALL_LOGIT_COL].iloc[0] == pytest.approx(
        XGBoostModel.logit(np.array([0.7]))[0]
    )


def test_pretrained_feature_signature_fails_on_extra_or_missing_baseball_features():
    with pytest.raises(ValueError, match="Extra in finetune"):
        validate_baseball_feature_signature(
            ["feature_a"],
            ["feature_a", "feature_b", MARKET_PROBABILITY_COL],
        )

    with pytest.raises(ValueError, match="Missing in finetune"):
        validate_baseball_feature_signature(
            ["feature_a", "feature_b"],
            ["feature_a", MARKET_PROBABILITY_COL],
        )


def test_baseball_feature_columns_excludes_market_and_pretrained_columns():
    assert baseball_feature_columns(
        [
            "feature_a",
            PRETRAINED_BASEBALL_LOGIT_COL,
            "p_open_home_median",
            "p_open_home_median_nv",
            "p_open_mean_nv_diff",
            "vig_open",
            "num_books",
            "logit_prob_home_std_nv",
            "home_opening_prob_raw",
            "away_opening_prob_nv",
            "home_opening_logit_temp",
            "feature_b",
        ]
    ) == ["feature_a", "feature_b"]

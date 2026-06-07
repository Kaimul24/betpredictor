import numpy as np
import pandas as pd
import pytest
import torch

from src.config import TwoHeadNNConfig
from src.data.models.model_comparison import (
    PredictionResult,
    assert_prediction_index,
    build_comparison_row,
    compare_predictions,
    load_two_head_model_from_checkpoint,
    market_baseline_result,
    prepare_xgboost_model_data_for_booster,
)
from src.data.models.two_head_nn import TwoHeadNN, save_checkpoint
from src.data.models.two_stage import (
    MARKET_PROBABILITY_COL,
    PRETRAINED_BASEBALL_LOGIT_COL,
    evaluate_probability_predictions,
    get_market_baseline_predictions,
)
from src.data.models.xgboost_model import XGBoostModel


def _model_data():
    index = pd.Index(["game-1", "game-2"], name="game_id")
    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            MARKET_PROBABILITY_COL: [0.4, 0.6],
        },
        index=index,
    )
    y = pd.DataFrame({"is_winner_home": [0, 1]}, index=index)
    return {
        "X_train": X.copy(),
        "y_train": y.copy(),
        "X_test": X.copy(),
        "y_test": y.copy(),
        "X_val": X.copy(),
        "y_val": y.copy(),
    }


def test_market_baseline_predictions_clip_probabilities():
    data = _model_data()
    data["X_test"][MARKET_PROBABILITY_COL] = [0.0, 1.0]

    predictions = get_market_baseline_predictions(data, split="test", eps=1e-4)

    assert predictions.tolist() == pytest.approx([1e-4, 1 - 1e-4])


def test_market_baseline_predictions_validate_split_and_column():
    data = _model_data()

    with pytest.raises(KeyError, match="X_missing"):
        get_market_baseline_predictions(data, split="missing")

    del data["X_test"][MARKET_PROBABILITY_COL]
    with pytest.raises(ValueError, match=MARKET_PROBABILITY_COL):
        get_market_baseline_predictions(data, split="test")


def test_evaluate_probability_predictions_returns_expected_metrics():
    metrics = evaluate_probability_predictions([0, 1], [0.25, 0.75])

    assert metrics["log_loss"] == pytest.approx(-np.log(0.75))
    assert metrics["brier"] == pytest.approx(0.0625)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_market_baseline_row_generation_from_model_data():
    data = _model_data()
    result = market_baseline_result(data, split="test")
    market_metrics = evaluate_probability_predictions(data["y_test"], result.predictions)

    row = build_comparison_row(result, data["y_test"], market_metrics)

    assert row["model"] == "market_baseline"
    assert row["n"] == 2
    assert row["delta_log_loss_vs_market"] == pytest.approx(0.0)
    assert row["delta_brier_vs_market"] == pytest.approx(0.0)


def test_comparison_table_computes_deltas_against_market():
    data = _model_data()
    baseline = market_baseline_result(data, split="test")
    model = PredictionResult(
        model="model",
        split="test",
        predictions=pd.Series([0.1, 0.9], index=data["X_test"].index),
        artifact="synthetic",
        calibrated=False,
    )

    comparison = compare_predictions(data, "test", [baseline, model])
    model_row = comparison[comparison["model"] == "model"].iloc[0]

    assert model_row["delta_log_loss_vs_market"] < 0
    assert model_row["delta_brier_vs_market"] < 0


class FakeFinalBooster:
    feature_names = ["feature_a", "feature_b", PRETRAINED_BASEBALL_LOGIT_COL]


class FakeBaseballBooster:
    feature_names = ["feature_a", "feature_b"]

    def predict(self, dmatrix):
        return np.full(dmatrix.num_row(), 0.7)


def test_xgboost_feature_preparation_adds_required_pretrained_logit():
    data = _model_data()

    updated = prepare_xgboost_model_data_for_booster(
        data,
        FakeFinalBooster(),
        baseball_booster=FakeBaseballBooster(),
    )

    assert PRETRAINED_BASEBALL_LOGIT_COL in updated["X_test"]
    assert PRETRAINED_BASEBALL_LOGIT_COL not in data["X_test"]
    assert updated["X_test"][PRETRAINED_BASEBALL_LOGIT_COL].tolist() == pytest.approx(
        XGBoostModel.logit(np.array([0.7, 0.7]))
    )


def test_two_head_checkpoint_config_instantiates_matching_architecture(tmp_path):
    config = TwoHeadNNConfig(
        training_mode="market_residual",
        stage="finetune",
        base_hidden_size=7,
        p_drop=0.0,
        max_residual=0.3,
    )
    model = TwoHeadNN(in_dim=2, base_hidden_size=7, p_drop=0.0, max_residual=0.3)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "best_model.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        train_loss=0.2,
        val_loss=0.3,
        path=checkpoint_path,
        config=config,
    )

    loaded_model, loaded_config = load_two_head_model_from_checkpoint(
        checkpoint_path,
        in_dim=2,
        device=torch.device("cpu"),
    )

    assert loaded_config.base_hidden_size == 7
    assert loaded_config.max_residual == pytest.approx(0.3)
    assert loaded_model.base_hidden_size == 7
    assert loaded_model.max_residual == pytest.approx(0.3)


def test_prediction_index_mismatch_raises_clear_error():
    result = PredictionResult(
        model="model",
        split="test",
        predictions=pd.Series([0.1, 0.2], index=pd.Index(["a", "b"])),
        artifact="synthetic",
        calibrated=False,
    )

    with pytest.raises(ValueError, match="canonical split index"):
        assert_prediction_index(result, pd.Index(["c", "d"]))

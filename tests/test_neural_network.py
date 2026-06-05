import pandas as pd
import pytest
import torch
from torch.utils.data import SequentialSampler

from src.config import NeuralNetworkConfig
from src.data.models.neural_network import prepare_mlp_data


def _model_data():
    index = pd.Index(range(6), name="row")
    X = pd.DataFrame(
        {
            "feature_order": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_other": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "p_open_home_median_nv": [0.45, 0.46, 0.47, 0.48, 0.49, 0.50],
        },
        index=index,
    )
    y = pd.DataFrame({"is_winner_home": [0, 1, 0, 1, 0, 1]}, index=index)

    return {
        "X_train": X.iloc[:4].copy(),
        "y_train": y.iloc[:4].copy(),
        "X_val": X.iloc[4:5].copy(),
        "y_val": y.iloc[4:5].copy(),
        "X_test": X.iloc[5:].copy(),
        "y_test": y.iloc[5:].copy(),
    }


def test_prepare_mlp_data_keeps_training_batches_in_input_order():
    config = NeuralNetworkConfig(stage="finetune", train_batch=2, val_batch=2)
    data = prepare_mlp_data(
        config=config,
        model_data=_model_data(),
    )

    assert isinstance(data["train_dl"].sampler, SequentialSampler)

    observed_order = []
    for X, _, _ in data["train_dl"]:
        observed_order.extend(X[:, 0].tolist())

    assert observed_order == [1.0, 2.0, 3.0, 4.0]


def test_prepare_mlp_data_uses_no_vig_market_probability_as_base_logit():
    config = NeuralNetworkConfig(stage="finetune", train_batch=2, val_batch=2)
    data = prepare_mlp_data(
        config=config,
        model_data=_model_data(),
    )

    _, base_logit, _ = next(iter(data["train_dl"]))

    assert torch.sigmoid(base_logit.squeeze()).tolist() == pytest.approx([0.45, 0.46])

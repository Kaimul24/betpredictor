import argparse
import json
import logging
import sys

import pytest
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TwoHeadNNConfig
import src.data.models.two_head_nn as two_head_module
from src.data.models.two_head_nn import (
    DatasetFineTune,
    TwoHeadNN,
    apply_pretrain_hyperparameters,
    create_args,
    load_checkpoint,
    load_pretrained_checkpoint,
    load_pretrain_hyperparameters,
    pretrain,
    pretrain_stability_objective,
    save_checkpoint,
    train,
    tune_pretrain_hyperparameters,
    validate_tuning_mode,
)


LOGGER = logging.getLogger("test_two_head_nn")


def test_two_head_checkpoint_uses_weights_only_safe_config_dict(tmp_path):
    config = TwoHeadNNConfig(training_mode="baseball_only", stage="pretrain", epochs=1)
    model = TwoHeadNN(in_dim=2, base_hidden_size=4, p_drop=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        train_loss=0.2,
        val_loss=0.3,
        path=checkpoint_path,
        config=config,
    )

    raw_checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert raw_checkpoint["config"] == config.to_dict()
    assert "args" not in raw_checkpoint

    loaded_model = TwoHeadNN(in_dim=2, base_hidden_size=4, p_drop=0.0)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=config.lr)

    model_out, optimizer_out = load_checkpoint(
        model=loaded_model,
        optimizer=loaded_optimizer,
        model_path=checkpoint_path,
        config=config,
    )

    assert model_out is loaded_model
    assert optimizer_out is loaded_optimizer


def test_two_head_load_pretrained_checkpoint_from_disk(tmp_path):
    pretrain_config = TwoHeadNNConfig(training_mode="baseball_only", stage="pretrain", epochs=1)
    finetune_config = TwoHeadNNConfig(training_mode="market_residual", stage="finetune", epochs=1)
    pretrained_model = TwoHeadNN(in_dim=2, base_hidden_size=4, p_drop=0.0)
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=pretrain_config.lr)
    checkpoint_path = tmp_path / "pretrained.pt"

    save_checkpoint(
        model=pretrained_model,
        optimizer=optimizer,
        epoch=0,
        train_loss=0.2,
        val_loss=0.3,
        path=checkpoint_path,
        config=pretrain_config,
    )

    finetune_model = TwoHeadNN(in_dim=2, base_hidden_size=4, p_drop=0.0)
    loaded_model = load_pretrained_checkpoint(finetune_model, checkpoint_path)

    assert loaded_model is finetune_model
    assert pretrain_config.to_dict() != finetune_config.to_dict()
    for expected, actual in zip(pretrained_model.parameters(), finetune_model.parameters()):
        torch.testing.assert_close(expected, actual)


def test_finetune_dataset_uses_baseball_features_and_market_baseline():
    X = pd.DataFrame(
        {
            "baseball_a": [1.0, 2.0],
            "baseball_b": [3.0, 4.0],
            "p_open_home_median_nv": [0.55, 0.60],
            "num_books": [8, 7],
            "vig_open": [0.03, 0.04],
            "logit_prob_home_std_nv": [0.1, 0.2],
        }
    )
    y = pd.Series([1, 0])

    dataset = DatasetFineTune(X, y)

    assert dataset.feature_columns == ["baseball_a", "baseball_b"]
    assert dataset.X.shape == (2, 2)
    assert dataset.p_mkt.tolist() == [0.55, 0.60]


def test_two_head_cli_parses_pretrain_tuning_fields(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "two_head_nn.py",
            "--retune",
            "--use-hyperparams",
            "--pretrain-trials",
            "3",
            "--weight-decay",
            "0.05",
        ],
    )

    args = create_args()
    config = TwoHeadNNConfig.from_namespace(argparse.Namespace(**vars(args), stage="finetune"))

    assert config.retune is True
    assert config.use_hyperparams is True
    assert config.pretrain_trials == 3
    assert config.weight_decay == pytest.approx(0.05)


def test_saved_hyperparams_apply_to_pretrain_config_without_mutating_finetune_config(tmp_path):
    hyperparam_path = tmp_path / "two_head_pretrain_hyperparams.json"
    hyperparam_path.write_text(
        json.dumps(
            {
                "best_params": {
                    "base_hidden_size": 64,
                    "p_drop": 0.35,
                    "lr": 0.0007,
                    "weight_decay": 0.02,
                },
                "best_value": 0.67,
                "n_trials": 2,
            }
        ),
        encoding="utf-8",
    )
    config = TwoHeadNNConfig(
        training_mode="stacked",
        stage="finetune",
        base_hidden_size=32,
        p_drop=0.2,
        lr=1e-4,
        weight_decay=0.03,
    )

    params = load_pretrain_hyperparameters(hyperparam_path)
    pretrain_config = config.for_stage(
        training_mode="baseball_only",
        stage="pretrain",
        perspective_duplication=True,
    )
    tuned_pretrain_config = apply_pretrain_hyperparameters(pretrain_config, params)
    finetune_config = config.for_stage(
        training_mode="market_residual",
        stage="finetune",
        perspective_duplication=False,
    )

    assert tuned_pretrain_config.base_hidden_size == 64
    assert tuned_pretrain_config.p_drop == pytest.approx(0.35)
    assert tuned_pretrain_config.lr == pytest.approx(0.0007)
    assert tuned_pretrain_config.weight_decay == pytest.approx(0.02)
    assert finetune_config.base_hidden_size == 32
    assert finetune_config.p_drop == pytest.approx(0.2)
    assert finetune_config.lr == pytest.approx(1e-4)
    assert finetune_config.weight_decay == pytest.approx(0.03)


def test_tiny_optuna_pretrain_tuning_returns_all_hyperparameter_keys(tmp_path):
    torch.manual_seed(42)
    X = torch.randn(12, 3)
    y = (X[:, 0] > 0).float()
    train_loader = DataLoader(TensorDataset(X[:8], y[:8]), batch_size=4, shuffle=False)
    val_loader = DataLoader(TensorDataset(X[8:], y[8:]), batch_size=4, shuffle=False)
    config = TwoHeadNNConfig(
        training_mode="baseball_only",
        stage="pretrain",
        epochs=1,
        pretrain_trials=2,
        base_hidden_size=16,
    )

    params = tune_pretrain_hyperparameters(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        in_dim=3,
        device=torch.device("cpu"),
        logger=LOGGER,
        n_trials=2,
        hyperparam_path=tmp_path / "params.json",
    )

    assert set(params) == {"base_hidden_size", "p_drop", "lr", "weight_decay"}


def test_pretrain_stability_objective_penalizes_late_validation_drift():
    val_losses = [0.70, 0.68, 0.67, 0.69, 0.72, 0.74]

    objective = pretrain_stability_objective(val_losses)

    best_smoothed_val = (0.70 + 0.68 + 0.67 + 0.69 + 0.72) / 5
    overfit_penalty = ((0.68 + 0.67 + 0.69 + 0.72 + 0.74) / 5) - 0.67
    expected = best_smoothed_val + 0.25 * overfit_penalty
    assert objective == pytest.approx(expected)


def test_pretrain_stability_objective_handles_short_trials():
    assert pretrain_stability_objective([0.7]) == pytest.approx(0.7)


def test_train_runs_full_epoch_count_and_saves_best_checkpoint(tmp_path):
    config = TwoHeadNNConfig(
        training_mode="baseball_only",
        stage="pretrain",
        epochs=3,
        base_hidden_size=4,
        p_drop=0.0,
    )
    model = TwoHeadNN(in_dim=2, base_hidden_size=4, p_drop=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    calls = {"train": 0, "val": 0, "scheduler": 0}
    val_losses = [0.5, 0.4, 0.45]

    class Scheduler:
        def step(self):
            calls["scheduler"] += 1

    def train_epoch_fn(*args):
        calls["train"] += 1
        return float(calls["train"])

    def val_epoch_fn(*args):
        val_loss = val_losses[calls["val"]]
        calls["val"] += 1
        return val_loss

    train_losses, observed_val_losses, best_val, best_epoch = train(
        model=model,
        train_loader=[],
        val_loader=[],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=Scheduler(),
        device=torch.device("cpu"),
        train_epoch_fn=train_epoch_fn,
        val_epoch_fn=val_epoch_fn,
        checkpoint_dir=tmp_path,
        epochs=config.epochs,
        logger=LOGGER,
        config=config,
    )

    assert calls == {"train": 3, "val": 3, "scheduler": 3}
    assert train_losses == [1.0, 2.0, 3.0]
    assert observed_val_losses == val_losses
    assert best_val == pytest.approx(0.4)
    assert best_epoch == 1
    best_checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=True)
    assert best_checkpoint["epoch"] == 1


def test_pretrain_reloads_best_checkpoint_before_returning(monkeypatch, tmp_path):
    config = TwoHeadNNConfig(
        training_mode="baseball_only",
        stage="pretrain",
        epochs=2,
        pretrain_dir=tmp_path,
        base_hidden_size=4,
        p_drop=0.0,
    )
    calls = {}

    def fake_prepare_data(*args, **kwargs):
        return [], [], [], 2

    def fake_train(*args, **kwargs):
        return [0.5, 0.4], [0.7, 0.6], 0.6, 1

    def fake_plot_loss(*args, **kwargs):
        return None

    def fake_load_pretrained_checkpoint(model, model_path, device=None):
        calls["path"] = model_path
        model.loaded_from_best_checkpoint = True
        return model

    monkeypatch.setattr(two_head_module, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(two_head_module, "train", fake_train)
    monkeypatch.setattr(two_head_module, "_plot_loss", fake_plot_loss)
    monkeypatch.setattr(two_head_module, "load_pretrained_checkpoint", fake_load_pretrained_checkpoint)

    model = pretrain(torch.device("cpu"), LOGGER, config)

    assert calls["path"] == tmp_path / "best_model.pt"
    assert model.loaded_from_best_checkpoint is True


def test_market_residual_retune_raises_pretrain_only_error():
    config = TwoHeadNNConfig(training_mode="market_residual", stage="finetune", retune=True)

    with pytest.raises(ValueError, match="baseball-only pretrain"):
        validate_tuning_mode(config)

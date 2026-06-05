import torch
import pandas as pd

from src.config import TwoHeadNNConfig
from src.data.models.two_head_nn import (
    DatasetFineTune,
    TwoHeadNN,
    load_checkpoint,
    load_pretrained_checkpoint,
    save_checkpoint,
)


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

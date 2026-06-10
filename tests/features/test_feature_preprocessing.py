from unittest.mock import Mock

import pandas as pd
import pytest

from src.config import FeatureConfig, TwoHeadNNConfig
from src.data.features.feature_preprocessing import PreProcessing


def _args(training_mode):
    return FeatureConfig(
        training_mode=training_mode,
        stage="pretrain" if training_mode == "baseball_only" else "finetune",
        model_type="xgboost",
        batter_halflives=(4, 12),
        starter_halflives=(3, 8),
        reliever_halflives=(3, 8),
        team_halflives=(3, 8, 20),
    )


def _season_frame(season, dates):
    rows = []
    for idx, game_date in enumerate(pd.to_datetime(dates)):
        rows.append(
            {
                "season": season,
                "game_date": game_date,
                "dh": 0,
                "game_datetime": game_date + pd.Timedelta(hours=19),
                "home_team": f"H{idx}",
                "away_team": f"A{idx}",
                "game_id": f"{season}-{idx}",
                "is_winner_home": idx % 2,
                "feature": float(idx),
            }
        )
    return pd.DataFrame(rows).set_index(
        ["season", "game_date", "dh", "game_datetime", "home_team", "away_team", "game_id"]
    )


def test_preprocessing_split_uses_final_season_for_val_test_cutoff():
    processor = PreProcessing(
        PreProcessing.PRETRAIN_YEARS,
        config=_args("baseball_only"),
    )
    processor.logger = Mock()

    filtered_dfs = [
        _season_frame(2016, ["2016-04-01", "2016-04-02"]),
        _season_frame(2017, ["2017-04-01", "2017-04-02"]),
        _season_frame(2018, ["2018-04-01", "2018-04-02"]),
        _season_frame(2019, ["2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04"]),
    ]

    train_data, val_df, test_df = processor._split_data(filtered_dfs)

    assert train_data.index.get_level_values("season").tolist() == [2016, 2016, 2017, 2017, 2018, 2018]
    assert val_df.index.get_level_values("game_date").tolist() == [
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2019-04-02"),
        pd.Timestamp("2019-04-03"),
    ]
    assert test_df.index.get_level_values("game_date").tolist() == [pd.Timestamp("2019-04-04")]


def test_preprocessing_requires_feature_config():
    with pytest.raises(TypeError, match="FeatureConfig"):
        PreProcessing(
            PreProcessing.PRETRAIN_YEARS,
            config=None,
        )


def test_preprocessing_uses_dataclass_config_without_mutating_it():
    config = _args("baseball_only")
    processor = PreProcessing(
        PreProcessing.PRETRAIN_YEARS,
        config=config,
    )

    assert processor.training_mode == "baseball_only"
    assert processor.args == config


def test_preprocessing_cache_key_uses_stable_dataclass_values():
    config = _args("market_residual")

    processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=config,
    )

    assert processor._cache_key() == (
        "xgboost_finetune_market_residual_seasons-2021_2022_2023_2024_2025_"
        "perspective-0_structured-0_structured_player_v1_bat-4-12_"
        "sp-3-8_rp-3-8_team-3-8-20"
    )


def test_structured_player_features_config_flows_to_feature_config_and_cache_key():
    nn_config = TwoHeadNNConfig(
        training_mode="stacked",
        stage="finetune",
        structured_player_features=True,
    )

    feature_config = nn_config.to_feature_config(
        stage="pretrain",
        training_mode="baseball_only",
        model_type="mlp",
    )

    assert feature_config.structured_player_features is True

    structured_processor = PreProcessing(
        PreProcessing.PRETRAIN_YEARS,
        config=feature_config,
    )
    default_processor = PreProcessing(
        PreProcessing.PRETRAIN_YEARS,
        config=FeatureConfig(
            training_mode="baseball_only",
            stage="pretrain",
            model_type="mlp",
            structured_player_features=False,
        ),
    )

    assert "structured-1" in structured_processor._cache_key()
    assert "structured-0" in default_processor._cache_key()
    assert structured_processor._cache_key() != default_processor._cache_key()


def test_preprocessing_config_defaults_still_work():
    config = FeatureConfig(stage="finetune", model_type="xgboost")
    processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=config,
    )

    assert processor.training_mode == "market_residual"
    assert processor.stage == "finetune"
    assert processor.args.team_halflives == (3, 8, 20)

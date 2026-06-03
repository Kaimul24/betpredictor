from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from src.data.features.feature_preprocessing import PreProcessing


def _args(training_mode):
    return SimpleNamespace(
        training_mode=training_mode,
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
        model_type="xgboost",
        args=_args("baseball_only"),
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


def test_preprocessing_accepts_feature_mode_alias():
    args = SimpleNamespace(
        feature_mode="baseball_only",
        batter_halflives=(4, 12),
        starter_halflives=(3, 8),
        reliever_halflives=(3, 8),
        team_halflives=(3, 8, 20),
    )

    processor = PreProcessing(
        PreProcessing.PRETRAIN_YEARS,
        model_type="xgboost",
        args=args,
    )

    assert processor.training_mode == "baseball_only"
    assert processor.args.feature_mode == "baseball_only"
    assert not hasattr(args, "training_mode")


def test_preprocessing_does_not_mutate_training_mode_args():
    args = _args("market_residual")

    processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        model_type="xgboost",
        args=args,
    )

    assert processor.training_mode == "market_residual"
    assert processor.args.feature_mode == "market_residual"
    assert not hasattr(args, "feature_mode")


def test_preprocessing_default_args_still_work():
    processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        model_type="xgboost",
        args=None,
    )

    assert processor.training_mode == "market_residual"
    assert processor.stage == "finetune"
    assert processor.args.team_halflives == (3, 8, 20)

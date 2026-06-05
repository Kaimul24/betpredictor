from unittest.mock import Mock

import pandas as pd
import pytest

from src.config import FeatureConfig
from src.data.features.feature_preprocessing import PreProcessing
from src.data.features.perspective import (
    RELIEVER_COLS,
    STARTER_COLS,
    STARTER_VS_BATTER_COLS,
    TEAM_METRIC_COLS,
    duplicate_training_perspective,
)


INDEX_COLS = ["season", "game_date", "dh", "game_datetime", "home_team", "away_team", "game_id"]


def _config(*, perspective_duplication: bool) -> FeatureConfig:
    return FeatureConfig(
        training_mode="market_residual",
        stage="finetune",
        model_type="xgboost",
        perspective_duplication=perspective_duplication,
        batter_halflives=(),
        starter_halflives=(),
        reliever_halflives=(),
        team_halflives=(),
    )


def _perspective_row(
    *,
    season: int = 2021,
    game_date: str = "2021-04-01",
    game_id: str = "game-1",
    is_winner_home: int = 1,
    p_open_home_median_nv: float = 0.65,
) -> dict:
    game_date_ts = pd.Timestamp(game_date)
    row = {
        "season": season,
        "game_date": game_date_ts,
        "dh": 0,
        "game_datetime": game_date_ts + pd.Timedelta(hours=19),
        "home_team": f"HOME-{game_id}",
        "away_team": f"AWAY-{game_id}",
        "game_id": game_id,
        "is_winner_home": is_winner_home,
        "p_open_home_median_nv": p_open_home_median_nv,
        "homegrown_metric": 10.0,
        "awayness_metric": 20.0,
    }

    for i, col in enumerate(STARTER_COLS, start=1):
        row[f"home_{col}_season"] = 3.0 + i / 10
        row[f"away_{col}_season"] = 4.0 + i / 10

    for i, col in enumerate(RELIEVER_COLS, start=1):
        row[f"home_{col}_season"] = 5.0 + i / 10
        row[f"away_{col}_season"] = 6.0 + i / 10

    for i, col in enumerate(TEAM_METRIC_COLS, start=1):
        row[f"home_{col}_season"] = 0.40 + i / 100
        row[f"away_{col}_season"] = 0.50 + i / 100

    for i, col in enumerate(["starter_bb_percent"]):
        row[f"home_{col}_season"] = 2.0 + i / 10
        row[f"away_{col}_season"] = 2.5 + i / 10

    for i, col in enumerate(["woba", "k_percent", "bb_percent", "barrel_percent"]):
        row[f"home_{col}_season"] = 0.300 + i / 100
        row[f"away_{col}_season"] = 0.250 + i / 100

    row["home_starter_fip_season"] = 3.70
    row["away_starter_fip_season"] = 4.10
    row["home_woba_season"] = 0.330
    row["away_woba_season"] = 0.300
    row["home_away_starter_fip_woba_season_diff"] = 999.0
    return row


def _season_frame(season: int, dates: list[str]) -> pd.DataFrame:
    rows = [
        _perspective_row(
            season=season,
            game_date=game_date,
            game_id=f"{season}-{idx}",
            is_winner_home=idx % 2,
            p_open_home_median_nv=0.55 + idx / 100,
        )
        for idx, game_date in enumerate(dates)
    ]
    return pd.DataFrame(rows).set_index(INDEX_COLS)


def test_duplicate_training_perspective_adds_home_and_flipped_focal_rows():
    frame = pd.DataFrame([_perspective_row()])

    result = duplicate_training_perspective(
        frame,
        batter_halflives=(),
        starter_halflives=(),
        reliever_halflives=(),
        team_halflives=(),
    )

    assert len(result) == 2

    original = result.iloc[0]
    flipped = result.iloc[1]

    assert original["focal_is_home"] == 1
    assert original["is_winner_home"] == 1
    assert original["p_open_home_median_nv"] == pytest.approx(0.65)
    assert original["home_team"] == "HOME-game-1"
    assert original["away_team"] == "AWAY-game-1"

    assert flipped["focal_is_home"] == 0
    assert flipped["is_winner_home"] == 0
    assert flipped["p_open_home_median_nv"] == pytest.approx(0.35)
    assert flipped["home_team"] == "AWAY-game-1"
    assert flipped["away_team"] == "HOME-game-1"
    assert flipped["home_starter_fip_season"] == pytest.approx(4.10)
    assert flipped["away_starter_fip_season"] == pytest.approx(3.70)
    assert flipped["homegrown_metric"] == pytest.approx(10.0)
    assert flipped["awayness_metric"] == pytest.approx(20.0)


def test_duplicate_training_perspective_recomputes_asymmetric_matchup_columns_after_swap():
    frame = pd.DataFrame([_perspective_row()])

    result = duplicate_training_perspective(
        frame,
        batter_halflives=(),
        starter_halflives=(),
        reliever_halflives=(),
        team_halflives=(),
    )

    original = result.iloc[0]
    flipped = result.iloc[1]

    assert original["home_away_starter_fip_woba_season_diff"] == pytest.approx(3.70 - 0.300)
    assert flipped["home_away_starter_fip_woba_season_diff"] == pytest.approx(4.10 - 0.330)
    assert flipped["home_away_starter_fip_woba_season_diff"] != pytest.approx(
        -original["home_away_starter_fip_woba_season_diff"]
    )


def test_preprocessing_applies_perspective_duplication_only_to_training_split():
    processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=_config(perspective_duplication=True),
    )
    processor.logger = Mock()

    processed = processor._feature_scaling(
        [
            _season_frame(2021, ["2021-04-01"]),
            _season_frame(2022, ["2022-04-01"]),
            _season_frame(2023, ["2023-04-01"]),
            _season_frame(2024, ["2024-04-01"]),
            _season_frame(2025, ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04"]),
        ]
    )

    assert len(processed["X_train"]) == 8
    assert processed["X_train"]["focal_is_home"].tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    assert processed["y_train"]["is_winner_home"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert processed["X_train"]["p_open_home_median_nv"].tolist() == pytest.approx(
        [0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45]
    )

    assert len(processed["X_val"]) == 3
    assert len(processed["X_test"]) == 1
    assert processed["X_val"]["focal_is_home"].tolist() == [1, 1, 1]
    assert processed["X_test"]["focal_is_home"].tolist() == [1]
    assert processed["y_val"]["is_winner_home"].tolist() == [0, 1, 0]
    assert processed["y_test"]["is_winner_home"].tolist() == [1]
    assert processed["X_val"]["p_open_home_median_nv"].tolist() == pytest.approx([0.55, 0.56, 0.57])
    assert processed["X_test"]["p_open_home_median_nv"].tolist() == pytest.approx([0.58])


def test_preprocessing_cache_key_changes_when_perspective_duplication_is_enabled():
    default_processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=_config(perspective_duplication=False),
    )
    duplicated_processor = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=_config(perspective_duplication=True),
    )

    assert duplicated_processor._cache_key() != default_processor._cache_key()

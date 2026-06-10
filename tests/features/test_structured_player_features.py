import pandas as pd
import pytest

from src.data.features.player_features.structured import StructuredPlayerFeatures


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["game1"],
            "game_date": [pd.Timestamp("2024-04-02")],
            "dh": [0],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
        }
    ).set_index(["game_id", "game_date", "dh", "home_team", "away_team"])


def test_lineup_slot_features_are_fixed_width_and_use_exact_rolling_rows():
    lineups = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-04-02", "2024-04-02", "2024-04-02"]),
            "dh": [0, 0, 0],
            "team": ["BOS", "BOS", "NYY"],
            "opposing_team": ["NYY", "NYY", "BOS"],
            "player_id": [10, 99, 20],
            "position": ["CF", "SS", "RF"],
            "batting_order": [1, 2, 1],
            "season": [2024, 2024, 2024],
        }
    )
    batting_features = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-04-02", "2024-04-02", "2024-04-03"]),
            "dh": [0, 0, 0],
            "player_id": [10, 20, 10],
            "woba_season": [0.310, 0.360, 0.999],
            "woba_ewm_h3": [0.320, 0.370, 0.999],
            "frv_per_9": [1.5, -0.5, 99.0],
        }
    )

    built = StructuredPlayerFeatures(_schedule(), batter_halflives=(3,)).lineup_slot_result(
        lineups,
        batting_features,
    )
    result = built.data
    row = result.iloc[0]

    assert built.index.equals(result.index)
    assert built.feature_names == result.columns.to_list()
    assert built.metadata["kind"] == "lineup_slots"
    assert built.metadata["slots"] == list(range(1, 10))

    for slot in range(1, 10):
        assert f"away_batter_{slot}_woba_season" in result.columns
        assert f"home_batter_{slot}_woba_season" in result.columns
        assert f"away_batter_{slot}_is_missing_player" in result.columns
        assert f"home_batter_{slot}_is_missing_player" in result.columns

    assert row["away_batter_1_woba_season"] == pytest.approx(0.310)
    assert row["away_batter_1_woba_ewm_h3"] == pytest.approx(0.320)
    assert row["home_batter_1_frv_per_9"] == pytest.approx(-0.5)
    assert row["away_batter_1_woba_season"] != pytest.approx(0.999)
    assert not row["away_batter_1_is_missing_player"]
    assert not row["home_batter_1_is_missing_player"]

    assert pd.isna(row["away_batter_2_woba_season"])
    assert row["away_batter_2_is_missing_player"]
    assert pd.isna(row["away_batter_3_woba_season"])
    assert row["away_batter_3_is_missing_player"]


def test_starter_features_build_home_away_vectors_from_team_starter_columns_only():
    pitching_features = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-04-02", "2024-04-02"]),
            "dh": [0, 0],
            "team": ["BOS", "NYY"],
            "opposing_team": ["NYY", "BOS"],
            "team_starter_player_id": [10, 20],
            "team_starter_fip_season": [3.75, 4.10],
            "team_starter_fip_ewm_h8": [3.55, 4.25],
            "team_pen_fip_season": [3.20, 3.90],
            "opposing_team_starter_fip_season": [4.10, 3.75],
        }
    )

    built = StructuredPlayerFeatures(_schedule(), starter_halflives=(8,)).starter_result(
        pitching_features,
    )
    result = built.data
    row = result.iloc[0]

    assert built.feature_names == result.columns.to_list()
    assert built.metadata["kind"] == "starter_vectors"
    assert built.metadata["source"] == "team_starter_rows"
    assert row["away_starter_vec_fip_season"] == pytest.approx(3.75)
    assert row["home_starter_vec_fip_ewm_h8"] == pytest.approx(4.25)
    assert not row["away_starter_vec_is_missing_starter"]
    assert not row["home_starter_vec_is_missing_starter"]
    assert "away_starter_vec_player_id" not in result.columns
    assert "away_starter_vec_pen_fip_season" not in result.columns
    assert "away_starter_vec_opposing_team_starter_fip_season" not in result.columns


def test_starter_features_can_use_already_adjusted_side_columns():
    final_features = pd.DataFrame(
        {
            "home_starter_fip_season": [-0.25],
            "away_starter_fip_season": [0.15],
            "home_starter_fip_ewm_h8": [-0.10],
            "away_starter_fip_ewm_h8": [0.30],
            "home_pen_fip_season": [0.20],
        },
        index=_schedule().index,
    )

    built = StructuredPlayerFeatures(_schedule(), starter_halflives=(8,)).starter_result(final_features)
    row = built.data.iloc[0]

    assert built.metadata["source"] == "side_columns"
    assert row["home_starter_vec_fip_season"] == pytest.approx(-0.25)
    assert row["away_starter_vec_fip_ewm_h8"] == pytest.approx(0.30)
    assert not row["home_starter_vec_is_missing_starter"]
    assert "home_starter_vec_pen_fip_season" not in built.data.columns

"""Tests for PitchingFeatures covering starter, bullpen, and usage logic."""

from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from src.data.features.player_features import pitching
from src.data.features.player_features.pitching import PitchingFeatures
from src.utils import normalize_names
from tests.conftest import insert_rosters


def _make_pitching_row(
    player_id: int,
    mlb_id: int,
    name: str,
    team: str,
    game_date: str,
    dh: int,
    games: int,
    gs: int,
    era: float,
    babip: float,
    ip: float,
    tbf: int,
    bip: int,
    runs: int,
    k_percent: float,
    bb_percent: float,
    barrel_percent: float,
    hard_hit: float,
    ev: float,
    hr_fb: float,
    siera: float,
    fip: float,
    stuff: int,
    iffb: int,
    wpa: float,
    gmli: float,
    fa_percent: float,
    fc_percent: float,
    si_percent: float,
    fa_velo: float,
    fc_velo: float,
    si_velo: float,
) -> dict:
    return {
        "player_id": player_id,
        "mlb_id": mlb_id,
        "name": name,
        "normalized_player_name": normalize_names(name),
        "game_date": game_date,
        "team": team,
        "dh": dh,
        "games": games,
        "gs": gs,
        "era": era,
        "babip": babip,
        "ip": ip,
        "tbf": tbf,
        "bip": bip,
        "runs": runs,
        "k_percent": k_percent,
        "bb_percent": bb_percent,
        "barrel_percent": barrel_percent,
        "hard_hit": hard_hit,
        "ev": ev,
        "hr_fb": hr_fb,
        "siera": siera,
        "fip": fip,
        "stuff": stuff,
        "iffb": iffb,
        "wpa": wpa,
        "gmli": gmli,
        "fa_percent": fa_percent,
        "fc_percent": fc_percent,
        "si_percent": si_percent,
        "fa_velo": fa_velo,
        "fc_velo": fc_velo,
        "si_velo": si_velo,
        "season": 2024,
    }


@pytest.fixture
def pitching_stats_dataframe(clean_db):
    rosters = [
        ("2024-03-27", 2024, "TEX", "Rick Ace", 1001, "P", "Active"),
        ("2024-03-27", 2024, "TEX", "Tom Relief", 1101, "P", "Active"),
        ("2024-03-27", 2024, "TEX", "Sam Setup", 1102, "P", "Active"),
        ("2024-03-27", 2024, "SEA", "Opie Penner", 2001, "P", "Active"),
        ("2024-03-27", 2024, "SEA", "Ray Lefty", 2101, "P", "Active"),
        ("2024-03-27", 2024, "SEA", "Ned Closer", 2102, "P", "Active"),
        ("2024-04-01", 2024, "TEX", "Rick Ace", 1001, "P", "Active"),
        ("2024-04-01", 2024, "TEX", "Tom Relief", 1101, "P", "Active"),
        ("2024-04-01", 2024, "TEX", "Sam Setup", 1102, "P", "Active"),
        ("2024-04-01", 2024, "SEA", "Opie Penner", 2001, "P", "Active"),
        ("2024-04-01", 2024, "SEA", "Ray Lefty", 2101, "P", "Active"),
        ("2024-04-01", 2024, "SEA", "Ned Closer", 2102, "P", "Active"),
    ]
    insert_rosters(clean_db, rosters)

    lineup_entries = [
        ("2024-03-27", 1, "TEX", 0, 2, "SEA", 1001, 2001, 2024),
        ("2024-03-27", 2, "SEA", 0, 1, "TEX", 2001, 1001, 2024),
        ("2024-04-01", 1, "TEX", 0, 2, "SEA", 1001, 2001, 2024),
        ("2024-04-01", 2, "SEA", 0, 1, "TEX", 2001, 1001, 2024),
    ]

    lineup_query = (
        "INSERT INTO lineups (game_date, team_id, team, dh, opposing_team_id, opposing_team, "
        "team_starter_id, opposing_starter_id, season) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    clean_db.execute_many_write_queries(lineup_query, lineup_entries)

    rows = [
        _make_pitching_row(1001, 5001, "Rick Ace", "TEX", "2024-03-27", 0, 1, 1, 2.5, 0.282, 6.0, 24, 12,
                           2, 30.0, 8.0, 5.0, 34.0, 88.5, 12.0, 3.05, 3.10, 105, 2, 0.15, 1.10,
                           0.45, 0.25, 0.30, 95.1, 88.2, 91.5),
        _make_pitching_row(1001, 5001, "Rick Ace", "TEX", "2024-04-01", 0, 1, 1, 0.9, 0.265, 6.0, 24, 10,
                           0, 60.0, 2.0, 4.0, 20.0, 85.0, 5.0, 2.10, 2.40, 108, 1, 0.25, 1.05,
                           0.50, 0.20, 0.30, 95.7, 88.8, 92.0),
        _make_pitching_row(2001, 6001, "Opie Penner", "SEA", "2024-03-27", 0, 5, 1, 3.4, 0.295, 5.0, 22, 11,
                           3, 24.0, 9.0, 6.0, 36.0, 87.0, 13.0, 3.80, 3.65, 97, 1, -0.05, 0.95,
                           0.40, 0.35, 0.25, 93.0, 86.0, 90.5),
        _make_pitching_row(2001, 6001, "Opie Penner", "SEA", "2024-04-01", 0, 5, 1, 1.5, 0.275, 3.0, 16, 7,
                           1, 55.0, 5.0, 5.0, 22.0, 83.0, 7.0, 2.20, 2.50, 99, 1, 0.10, 0.90,
                           0.45, 0.30, 0.25, 94.2, 87.1, 91.2),
        _make_pitching_row(1101, 5101, "Tom Relief", "TEX", "2024-03-27", 0, 1, 0, 1.8, 0.310, 1.1, 5, 3,
                           0, 35.0, 5.0, 2.0, 20.0, 84.5, 4.0, 2.50, 2.70, 102, 0, 0.08, 1.30,
                           0.55, 0.25, 0.20, 96.0, 88.5, 92.5),
        _make_pitching_row(1101, 5101, "Tom Relief", "TEX", "2024-04-01", 0, 1, 0, 9.5, 0.335, 1.0, 5, 4,
                           4, 30.0, 12.0, 10.0, 55.0, 92.0, 20.0, 6.80, 5.90, 102, 0, -0.12, 1.40,
                           0.45, 0.30, 0.25, 96.4, 88.9, 92.8),
        _make_pitching_row(1102, 5102, "Sam Setup", "TEX", "2024-03-27", 0, 1, 0, 2.1, 0.300, 1.0, 4, 2,
                           0, 32.0, 7.0, 3.0, 28.0, 86.0, 6.0, 2.90, 3.05, 101, 0, 0.02, 0.80,
                           0.40, 0.30, 0.30, 95.2, 87.8, 91.7),
        _make_pitching_row(1102, 5102, "Sam Setup", "TEX", "2024-04-01", 0, 1, 0, 10.0, 0.340, 0.2, 3, 2,
                           3, 20.0, 15.0, 12.0, 60.0, 94.0, 25.0, 7.40, 6.80, 101, 0, -0.20, 0.75,
                           0.35, 0.30, 0.35, 95.5, 88.1, 92.0),
        _make_pitching_row(2101, 6101, "Ray Lefty", "SEA", "2024-03-27", 0, 1, 0, 2.4, 0.305, 1.2, 6, 3,
                           1, 28.0, 8.0, 5.0, 32.0, 85.0, 8.0, 3.10, 3.20, 99, 0, 0.04, 1.10,
                           0.48, 0.26, 0.26, 94.8, 87.5, 91.3),
        _make_pitching_row(2101, 6101, "Ray Lefty", "SEA", "2024-04-01", 0, 1, 0, 8.5, 0.330, 0.2, 3, 2,
                           3, 18.0, 14.0, 11.0, 58.0, 93.0, 22.0, 6.90, 6.35, 99, 0, -0.18, 1.05,
                           0.42, 0.28, 0.30, 95.1, 87.9, 91.8),
        _make_pitching_row(2102, 6102, "Ned Closer", "SEA", "2024-03-27", 0, 1, 0, 1.9, 0.290, 1.0, 4, 2,
                           0, 40.0, 6.0, 2.0, 18.0, 83.0, 4.0, 2.10, 2.30, 103, 0, 0.12, 1.70,
                           0.60, 0.25, 0.15, 97.0, 89.0, 93.1),
        _make_pitching_row(2102, 6102, "Ned Closer", "SEA", "2024-04-01", 0, 1, 0, 7.5, 0.325, 0.2, 3, 2,
                           2, 22.0, 16.0, 10.0, 52.0, 91.0, 18.0, 6.10, 5.70, 103, 0, -0.25, 1.80,
                           0.55, 0.30, 0.15, 97.5, 89.5, 93.5),
    ]

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def test_load_features_prevents_leakage_and_sets_flags(monkeypatch, tmp_path, pitching_stats_dataframe):
    monkeypatch.setattr(pitching, "FEATURES_CACHE_PATH", tmp_path)
    monkeypatch.setattr(pitching, "PITCHING_CACHE_PATH", "pitching_features_{}.parquet")

    features = PitchingFeatures(2024, pitching_stats_dataframe, force_recreate=True)
    result = features.load_features()

    assert not result.empty

    mask = (result["game_date"] == datetime(2024, 4, 1)) & (result["team"] == "TEX")
    row = result.loc[mask].iloc[0]

    assert row["team_starter_last_app_date"].date() == datetime(2024, 3, 27).date()
    assert bool(row["opposing_team_starter_is_opener"])

    k_bb = row["team_starter_k_bb_percent_season"]
    k_minus_bb = row["team_starter_k_percent_season"] - row["team_starter_bb_percent_season"]
    assert k_bb == pytest.approx(k_minus_bb, abs=0.5)
    assert abs(k_bb - row["team_starter_bb_percent_season"]) > 1.0

    assert row["team_pen_era_season"] < 5.0
    assert not np.isnan(row["team_pen_rest_days_mean"])
    assert 0.0 <= row["team_pen_freshness_gmliw"] <= 1.0

    assert not row.filter(like="team_pen_").isna().any()
    assert not row.filter(like="opposing_team_starter_").isna().any()


def test_compute_bullpen_usage_weighting():
    dummy = object.__new__(PitchingFeatures)

    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-04-01", "2024-04-01"]),
        "team": ["TEX", "TEX"],
        "last_app_date": pd.to_datetime(["2024-03-30", "2024-03-29"]),
        "gmli_ewm_h8": [2.0, 1.0],
    })

    usage = dummy._compute_bullpen_usage(df)
    assert len(usage) == 1
    row = usage.iloc[0]

    assert row["team_pen_rest_days_mean"] == pytest.approx(2.5, rel=1e-6)
    assert row["team_pen_rest_days_median"] == pytest.approx(2.5, rel=1e-6)
    assert row["team_pen_freshness_mean"] == pytest.approx(0.833333, rel=1e-6)
    assert row["team_pen_freshness_gmliw"] == pytest.approx(7.0 / 9.0, rel=1e-6)
    assert row["team_pen_hi_lev_available"] == pytest.approx(1.0, rel=1e-6)


def test_fill_bullpen_with_priors_replaces_missing():
    dummy = object.__new__(PitchingFeatures)

    df = pd.DataFrame({
        "team_pen_era_season": [np.nan, 3.2],
        "team_pen_k_percent_ewm_h3": [np.nan, 24.0],
        "team": ["TEX", "SEA"],
        "game_date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
    })

    priors = {"prior_era": 4.05, "prior_k_percent": 23.5}
    filled = dummy._fill_bullpen_with_priors(df, priors)

    assert filled.loc[0, "team_pen_era_season"] == pytest.approx(4.05, rel=1e-6)
    assert filled.loc[0, "team_pen_k_percent_ewm_h3"] == pytest.approx(23.5, rel=1e-6)
    assert filled.loc[1, "team_pen_era_season"] == pytest.approx(3.2, rel=1e-6)

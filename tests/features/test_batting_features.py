"""Tests for BattingFeatures aligned with the current rolling stats implementation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.features.base_feature import BaseFeatures
from src.data.features.player_features import batting
from src.data.features.player_features.batting import BattingFeatures


@pytest.fixture
def sample_batting_frame() -> pd.DataFrame:
    base_date = pd.Timestamp("2024-04-01")
    rows = []
    stats = [
        (0.30, 0.70, 110, 0.280, 0.26, 0.07, 0.28, 6.0, 35.0, 88.0, 0.18, 1.10, 0.10, 0.50, 0.03, 2, 4),
        (0.40, 0.75, 120, 0.300, 0.23, 0.08, 0.31, 7.0, 40.0, 89.0, 0.20, 1.05, 0.12, 0.80, 0.04, 3, 4),
        (0.50, 0.80, 130, 0.320, 0.21, 0.09, 0.34, 9.0, 45.0, 90.0, 0.22, 1.00, 0.15, 1.10, 0.05, 5, 4),
    ]

    for idx, (
        woba,
        ops,
        wrc_plus,
        babip,
        k_pct,
        bb_pct,
        bb_k,
        barrel,
        hard_hit,
        ev,
        iso,
        gb_fb,
        baserunning,
        wraa,
        wpa,
        bip,
        ab,
    ) in enumerate(stats):
        rows.append(
            {
                "player_id": "p1",
                "mlb_id": 101,
                "team": "NYY",
                "pos": "OF",
                "game_date": base_date + pd.Timedelta(days=idx),
                "dh": 0,
                "ab": ab,
                "pa": 4,
                "bip": bip,
                "woba": woba,
                "ops": ops,
                "wrc_plus": wrc_plus,
                "babip": babip,
                "k_percent": k_pct,
                "bb_percent": bb_pct,
                "bb_k": bb_k,
                "barrel_percent": barrel,
                "hard_hit": hard_hit,
                "ev": ev,
                "iso": iso,
                "gb_fb": gb_fb,
                "baserunning": baserunning,
                "wraa": wraa,
                "wpa": wpa,
                "season": 2024,
            }
        )

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@pytest.fixture
def fake_batting_cache(monkeypatch, tmp_path):
    store = {}

    def fake_to_parquet(self, path, index=True, *_, **__):
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        store[cache_path] = self.copy()
        cache_path.write_text("cached", encoding="utf-8")

    def fake_read_parquet(path, *_, **__):
        cache_path = Path(path)
        if cache_path not in store:
            raise FileNotFoundError(cache_path)
        return store[cache_path].copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(batting, "FEATURES_CACHE_PATH", tmp_path)
    monkeypatch.setattr(batting, "BATTING_CACHE_PATH", "batting_features_{}.parquet")

    return store


@pytest.fixture
def doubleheader_frame() -> pd.DataFrame:
    base_date = pd.Timestamp("2024-05-01")
    rows = [
        {
            "player_id": "p1",
            "mlb_id": 101,
            "team": "NYY",
            "pos": "OF",
            "game_date": base_date,
            "dh": 0,
            "ab": 4,
            "pa": 4,
            "bip": 3,
            "woba": 0.32,
            "ops": 0.72,
            "wrc_plus": 118,
            "babip": 0.285,
            "k_percent": 0.25,
            "bb_percent": 0.08,
            "bb_k": 0.30,
            "barrel_percent": 6.0,
            "hard_hit": 34.0,
            "ev": 87.5,
            "iso": 0.19,
            "gb_fb": 1.15,
            "baserunning": 0.09,
            "wraa": 0.40,
            "wpa": 0.02,
            "season": 2024,
        },
        {
            "player_id": "p1",
            "mlb_id": 101,
            "team": "NYY",
            "pos": "OF",
            "game_date": base_date,
            "dh": 1,
            "ab": 4,
            "pa": 4,
            "bip": 4,
            "woba": 0.38,
            "ops": 0.77,
            "wrc_plus": 126,
            "babip": 0.295,
            "k_percent": 0.22,
            "bb_percent": 0.085,
            "bb_k": 0.35,
            "barrel_percent": 7.5,
            "hard_hit": 38.0,
            "ev": 88.2,
            "iso": 0.21,
            "gb_fb": 1.05,
            "baserunning": 0.11,
            "wraa": 0.65,
            "wpa": 0.03,
            "season": 2024,
        },
        {
            "player_id": "p1",
            "mlb_id": 101,
            "team": "NYY",
            "pos": "OF",
            "game_date": base_date + pd.Timedelta(days=1),
            "dh": 0,
            "ab": 5,
            "pa": 5,
            "bip": 5,
            "woba": 0.43,
            "ops": 0.80,
            "wrc_plus": 132,
            "babip": 0.305,
            "k_percent": 0.20,
            "bb_percent": 0.09,
            "bb_k": 0.38,
            "barrel_percent": 8.5,
            "hard_hit": 42.0,
            "ev": 89.0,
            "iso": 0.23,
            "gb_fb": 0.98,
            "baserunning": 0.13,
            "wraa": 0.90,
            "wpa": 0.05,
            "season": 2024,
        },
    ]

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _shrink(prev_vals: pd.Series, prev_weights: pd.Series, prior: float, k: int) -> float:
    weight_total = float(prev_weights.sum())
    if weight_total <= 0:
        return float(prior)
    rate = float((prev_vals * prev_weights).sum() / weight_total)
    return (weight_total / (weight_total + k)) * rate + (k / (weight_total + k)) * prior


def test_load_features_returns_expected_columns(sample_batting_frame, fake_batting_cache):
    features = BattingFeatures(2024, sample_batting_frame, force_recreate=True)
    result = features.load_features()

    assert not result.empty

    expected_subset = {
        "player_id",
        "mlb_id",
        "team",
        "pos",
        "game_date",
        "dh",
        "season",
        "ab",
        "pa",
        "woba_season",
        "woba_ewm_h3",
        "woba_ewm_h10",
        "woba_ewm_h25",
        "last_app_date",
    }
    assert expected_subset.issubset(result.columns)

    ordered = result.sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)
    assert list(ordered["game_date"]) == sorted(sample_batting_frame["game_date"].tolist())


def test_season_stats_apply_prior_and_shift(sample_batting_frame, fake_batting_cache):
    features = BattingFeatures(2024, sample_batting_frame, force_recreate=True)
    result = features.load_features().sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)
    source = sample_batting_frame.sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)

    prior_woba = float((source["woba"] * source["pa"]).sum() / source["pa"].sum())
    assert result.loc[0, "woba_season"] == pytest.approx(prior_woba)
    assert pd.isna(result.loc[0, "last_app_date"])

    for idx in range(1, len(source)):
        expected = _shrink(source.iloc[:idx]["woba"], source.iloc[:idx]["pa"], prior_woba, 100)
        assert result.loc[idx, "woba_season"] == pytest.approx(expected)
        assert result.loc[idx, "last_app_date"] == source.loc[idx - 1, "game_date"]
        assert result.loc[idx, "woba_season"] != pytest.approx(source.loc[idx, "woba"])


def test_metric_specific_denominators(sample_batting_frame, fake_batting_cache):
    features = BattingFeatures(2024, sample_batting_frame, force_recreate=True)
    result = features.load_features().sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)
    source = sample_batting_frame.sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)

    barrel_prior = float((source["barrel_percent"] * source["bip"]).sum() / source["bip"].sum())
    iso_prior = float((source["iso"] * source["ab"]).sum() / source["ab"].sum())

    assert result.loc[0, "barrel_percent_season"] == pytest.approx(barrel_prior)
    assert result.loc[0, "iso_season"] == pytest.approx(iso_prior)

    expected_barrel = _shrink(source.iloc[:1]["barrel_percent"], source.iloc[:1]["bip"], barrel_prior, 70)
    expected_iso = _shrink(source.iloc[:1]["iso"], source.iloc[:1]["ab"], iso_prior, 100)

    assert result.loc[1, "barrel_percent_season"] == pytest.approx(expected_barrel)
    assert result.loc[1, "iso_season"] == pytest.approx(expected_iso)


def test_load_features_uses_cache_when_available(sample_batting_frame, fake_batting_cache, monkeypatch):
    features = BattingFeatures(2024, sample_batting_frame, force_recreate=False)
    first = features.load_features()

    cache_paths = list(fake_batting_cache.keys())
    assert cache_paths, "Expected first run to persist cache"
    assert cache_paths[0].exists()

    def fail_compute(*_, **__):
        raise AssertionError("compute_rolling_stats should not run when cache exists")

    monkeypatch.setattr(BaseFeatures, "compute_rolling_stats", staticmethod(fail_compute))

    cached_features = BattingFeatures(2024, sample_batting_frame, force_recreate=False)
    second = cached_features.load_features()

    pd.testing.assert_frame_equal(first, second)


def test_doubleheader_games_respect_temporal_order(doubleheader_frame, fake_batting_cache):
    features = BattingFeatures(2024, doubleheader_frame, force_recreate=True)
    result = features.load_features().sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)

    source = doubleheader_frame.sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)
    prior_woba = float((source["woba"] * source["pa"]).sum() / source["pa"].sum())

    assert result.loc[0, "woba_season"] == pytest.approx(prior_woba)
    assert pd.isna(result.loc[0, "last_app_date"])

    expected_after_game1 = _shrink(source.iloc[:1]["woba"], source.iloc[:1]["pa"], prior_woba, 100)
    assert result.loc[1, "woba_season"] == pytest.approx(expected_after_game1)
    assert result.loc[1, "last_app_date"] == source.loc[0, "game_date"]

    expected_after_game2 = _shrink(source.iloc[:2]["woba"], source.iloc[:2]["pa"], prior_woba, 100)
    assert result.loc[2, "woba_season"] == pytest.approx(expected_after_game2)
    assert result.loc[2, "last_app_date"] == source.loc[1, "game_date"]


def test_temporal_validation_with_unsorted_input(sample_batting_frame, fake_batting_cache):
    scrambled = sample_batting_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    features = BattingFeatures(2024, scrambled, force_recreate=True)
    result = features.load_features().sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)

    source = sample_batting_frame.sort_values(["player_id", "game_date", "dh"]).reset_index(drop=True)
    prior_woba = float((source["woba"] * source["pa"]).sum() / source["pa"].sum())

    assert result.loc[0, "woba_season"] == pytest.approx(prior_woba)
    for idx in range(1, len(source)):
        expected = _shrink(source.iloc[:idx]["woba"], source.iloc[:idx]["pa"], prior_woba, 100)
        assert result.loc[idx, "woba_season"] == pytest.approx(expected)
        assert result.loc[idx, "last_app_date"] == source.loc[idx - 1, "game_date"]
        assert result.loc[idx, "woba_season"] != pytest.approx(source.loc[idx, "woba"])

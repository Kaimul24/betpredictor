"""Structured lineup and starter feature vectors."""

from dataclasses import dataclass

import pandas as pd
from pandas.core.api import DataFrame


GAME_INDEX = ["game_id", "game_date", "dh", "home_team", "away_team"]
BATTER_SLOTS = range(1, 10)


@dataclass(frozen=True)
class StructuredFeatureResult:
    """Feature matrix plus lightweight metadata for structured player vectors."""

    index: pd.MultiIndex
    data: DataFrame
    feature_names: list[str]
    metadata: dict[str, object]


class StructuredPlayerFeatures:
    """Build fixed-width per-player game features from precomputed rolling stats."""

    def __init__(
        self,
        schedule: DataFrame,
        *,
        batter_halflives: tuple[int, ...] = (4, 12),
        starter_halflives: tuple[int, ...] = (3, 8),
    ) -> None:
        self.schedule = schedule
        self.batter_halflives = batter_halflives
        self.starter_halflives = starter_halflives

    def lineup_slot_features(self, lineups: DataFrame, batting_features: DataFrame) -> DataFrame:
        """Return fixed 1-9 batter slot columns for home and away teams."""
        return self.lineup_slot_result(lineups, batting_features).data

    def lineup_slot_result(self, lineups: DataFrame, batting_features: DataFrame) -> StructuredFeatureResult:
        """Return fixed 1-9 batter slot features with metadata."""
        games = self._games()
        stats_cols = self._batter_stats_cols(batting_features)
        index = pd.MultiIndex.from_frame(games[GAME_INDEX])

        if not stats_cols:
            data = pd.DataFrame(index=index)
            return StructuredFeatureResult(
                index=index,
                data=data,
                feature_names=[],
                metadata={"kind": "lineup_slots", "slots": list(BATTER_SLOTS), "stat_columns": []},
            )

        batter_stats = batting_features.copy()
        batter_stats["game_date"] = pd.to_datetime(batter_stats["game_date"])
        batter_stats["player_id"] = batter_stats["player_id"].astype("Int64")

        result = pd.DataFrame(index=index)
        lineups = self._prepare_lineups(lineups)

        for side in ("away", "home"):
            side_features = self._side_lineup_features(
                games,
                lineups,
                batter_stats,
                stats_cols,
                side,
            )
            result = result.join(side_features, how="left")

        return StructuredFeatureResult(
            index=index,
            data=result,
            feature_names=result.columns.to_list(),
            metadata={
                "kind": "lineup_slots",
                "sides": ["away", "home"],
                "slots": list(BATTER_SLOTS),
                "stat_columns": stats_cols,
                "missing_indicator": "is_missing_player",
            },
        )

    def starter_features(self, pitching_features: DataFrame) -> DataFrame:
        """Return one starter stat vector for each home/away team."""
        return self.starter_result(pitching_features).data

    def starter_result(self, pitching_features: DataFrame) -> StructuredFeatureResult:
        """Return home/away starter vectors with metadata."""
        games = self._games()
        stats_cols = self._starter_stats_cols(pitching_features, "team_starter_")
        index = pd.MultiIndex.from_frame(games[GAME_INDEX])

        if self._has_side_starter_columns(pitching_features):
            data = self._side_starter_features_from_wide(pitching_features, games)
            return StructuredFeatureResult(
                index=index,
                data=data,
                feature_names=data.columns.to_list(),
                metadata={
                    "kind": "starter_vectors",
                    "source": "side_columns",
                    "sides": ["away", "home"],
                    "missing_indicator": "is_missing_starter",
                },
            )

        raw = pitching_features.copy()
        raw["game_date"] = pd.to_datetime(raw["game_date"])
        result = pd.DataFrame(index=index)

        for side in ("away", "home"):
            team_col = f"{side}_team"
            opp_col = "home_team" if side == "away" else "away_team"
            side_stats = games.merge(
                raw[["game_date", "dh", "team", "opposing_team", *stats_cols]],
                left_on=["game_date", "dh", team_col, opp_col],
                right_on=["game_date", "dh", "team", "opposing_team"],
                how="left",
            )
            side_stats[f"{side}_starter_vec_is_missing_starter"] = side_stats[stats_cols].isna().all(axis=1)
            side_stats = side_stats.drop(columns=["team", "opposing_team"])
            side_stats = side_stats.rename(
                columns={col: f"{side}_starter_vec_{col.removeprefix('team_starter_')}" for col in stats_cols}
            )
            side_stats = side_stats.set_index(GAME_INDEX)
            result = result.join(side_stats, how="left")

        return StructuredFeatureResult(
            index=index,
            data=result,
            feature_names=result.columns.to_list(),
            metadata={
                "kind": "starter_vectors",
                "source": "team_starter_rows",
                "sides": ["away", "home"],
                "stat_columns": stats_cols,
                "missing_indicator": "is_missing_starter",
            },
        )

    def _games(self) -> DataFrame:
        games = self.schedule.reset_index()[GAME_INDEX].drop_duplicates().copy()
        games["game_date"] = pd.to_datetime(games["game_date"])
        return games

    def _prepare_lineups(self, lineups: DataFrame) -> DataFrame:
        prepared = lineups.copy()
        prepared["game_date"] = pd.to_datetime(prepared["game_date"])
        prepared["player_id"] = prepared["player_id"].astype("Int64")
        prepared["batting_order"] = pd.to_numeric(prepared["batting_order"], errors="coerce")
        prepared = prepared[prepared["batting_order"].between(1, 9)]
        return prepared

    def _side_lineup_features(
        self,
        games: DataFrame,
        lineups: DataFrame,
        batting_features: DataFrame,
        stats_cols: list[str],
        side: str,
    ) -> DataFrame:
        team_col = f"{side}_team"
        opp_col = "home_team" if side == "away" else "away_team"
        side_lineups = games.merge(
            lineups,
            left_on=["game_date", "dh", team_col, opp_col],
            right_on=["game_date", "dh", "team", "opposing_team"],
            how="left",
        )

        with_stats = side_lineups.merge(
            batting_features[["game_date", "dh", "player_id", *stats_cols]],
            on=["game_date", "dh", "player_id"],
            how="left",
        )

        result = pd.DataFrame(index=pd.MultiIndex.from_frame(games[GAME_INDEX]))
        for slot in BATTER_SLOTS:
            slot_rows = with_stats[with_stats["batting_order"] == slot]
            missing_col = f"{side}_batter_{slot}_is_missing_player"
            slot_features = slot_rows[GAME_INDEX + stats_cols].drop_duplicates(GAME_INDEX)
            slot_features[missing_col] = slot_features[stats_cols].isna().all(axis=1)
            slot_features = slot_features.rename(
                columns={col: f"{side}_batter_{slot}_{col}" for col in stats_cols}
            )
            slot_features = slot_features.set_index(GAME_INDEX)
            result = result.join(slot_features, how="left")
            result[missing_col] = result[missing_col].where(result[missing_col].notna(), True).astype(bool)

        return result

    def _batter_stats_cols(self, batting_features: DataFrame) -> list[str]:
        suffixes = ["_season", *[f"_ewm_h{hl}" for hl in self.batter_halflives]]
        return [
            col
            for col in batting_features.columns
            if any(suffix in col for suffix in suffixes) or col == "frv_per_9"
        ]

    def _starter_stats_cols(self, pitching_features: DataFrame, prefix: str) -> list[str]:
        suffixes = ["_season", *[f"_ewm_h{hl}" for hl in self.starter_halflives]]
        excluded_fragments = ("_id", "player_id", "name", "last_app_date")
        return [
            col
            for col in pitching_features.columns
            if col.startswith(prefix)
            and not any(fragment in col for fragment in excluded_fragments)
            and (
                any(suffix in col for suffix in suffixes)
                or pd.api.types.is_bool_dtype(pitching_features[col])
                or pd.api.types.is_numeric_dtype(pitching_features[col])
            )
        ]

    def _has_side_starter_columns(self, pitching_features: DataFrame) -> bool:
        return any(col.startswith("home_starter_") for col in pitching_features.columns) and any(
            col.startswith("away_starter_") for col in pitching_features.columns
        )

    def _side_starter_features_from_wide(self, pitching_features: DataFrame, games: DataFrame) -> DataFrame:
        wide = pitching_features.reset_index().copy()
        wide["game_date"] = pd.to_datetime(wide["game_date"])
        merged = games.merge(wide, on=GAME_INDEX, how="left")
        result = pd.DataFrame(index=pd.MultiIndex.from_frame(games[GAME_INDEX]))

        for side in ("away", "home"):
            prefix = f"{side}_starter_"
            stats_cols = self._starter_stats_cols(merged, prefix)
            side_features = merged[GAME_INDEX + stats_cols].copy()
            side_features[f"{side}_starter_vec_is_missing_starter"] = side_features[stats_cols].isna().all(axis=1)
            side_features = side_features.rename(
                columns={col: f"{side}_starter_vec_{col.removeprefix(prefix)}" for col in stats_cols}
            )
            side_features = side_features.set_index(GAME_INDEX)
            result = result.join(side_features, how="left")

        return result

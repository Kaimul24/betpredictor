#!/usr/bin/env python3
"""
Orchestrates the feature engineering process for all features. Applies normalization.
"""

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import logging
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

from torch import parse_type_comment

from src.config import PROJECT_ROOT

from src.data.features.base_feature import BaseFeatures
from src.data.features.game_features.context import GameContextFeatures
from src.data.features.game_features.odds import Odds
from src.data.features.player_features.batting import BattingFeatures
from src.data.features.player_features.fielding import FieldingFeatures
from src.data.features.player_features.pitching import PitchingFeatures
from src.data.features.player_features.war import WAR
from src.data.features.team_features.team_features import TeamFeatures

from src.data.loaders.game_loader import GameLoader
from src.data.loaders.odds_loader import OddsLoader
from src.data.loaders.player_loader import PlayerLoader
from src.data.loaders.team_loader import TeamLoader
from src.utils import setup_logging, TupleAction

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_pipeline.log"

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Feature engineering runner")
    parser.add_argument("--force-recreate", action="store_true", help="Recreate batting rolling features, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Console log level")
    parser.add_argument(
        "--file-log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="debug",
        help="File log level when --log is enabled")
    parser.add_argument(
        "--batter-halflives", 
        nargs='*',
        type=int,
        action=TupleAction,
        default=(3, 8, 20),
        help="EWM halflives for batting stats")
    parser.add_argument(
        "--starter-halflives",
        nargs='*',
        type=int,
        action=TupleAction,
        default=(3, 8, 20),
        help="EWM halflives for starting pitching stats")
    parser.add_argument(
        "--reliever-halflives", 
        nargs='*',
        type=int,
        action=TupleAction,
        default=(3, 8, 20),
        help="EWM halflives for starting pitching stats")
    parser.add_argument(
        "--team-halflives",
        nargs='*',
        type=int,
        action=TupleAction,
        default=(3, 8, 20),
        help="EWM halflives for team metric stats")

    args = parser.parse_args()
    return args

class FeaturePipeline:

    def __init__(self, season: int, args, logger: Optional[logging.Logger] = None):
        self.season = season
        self.cache = {}
        self.logger = logger or logging.getLogger("feature_pipeline")
        self.args = args

    def start_pipeline(self, force_recreate: bool = False, mkt_only: bool = False) -> Tuple[DataFrame, DataFrame] | DataFrame:
        self.logger.info("="*60)
        self.logger.info(f" Starting feature pipeline...")
        self.logger.info("="*60 + "\n")
        schedule_data = self._load_schedule_data()

        far_future_games = schedule_data[(schedule_data['game_datetime'].astype('datetime64[ns]') - schedule_data['game_date'].astype('datetime64[ns]')) > pd.Timedelta("2 Days")]

        self.logger.info(f" Far rescheduled games\n{far_future_games}")
        self.logger.info(f" Dropping rescheduled games that are far past original game_date")

        schedule_data = schedule_data.drop(far_future_games.index)
        
        idx = ["game_id", "game_date", "dh", "home_team", "away_team"]
        odds_data = self._load_odds_data()
        odds_feats = Odds(odds_data, self.season, mkt_only).load_features()
        odds_sch_matched, raw_odds_data = self._match_schedule_to_odds(schedule_data, odds_feats)
        odds_sch_matched = odds_sch_matched.reset_index().set_index(idx)
        
        position_player_feats = self._get_position_player_features(odds_sch_matched, force_recreate).reset_index().set_index(idx)
        pitching_feats = self._get_pitcher_features(odds_sch_matched, force_recreate).reset_index().set_index(idx)
        context_feats = GameContextFeatures(self.season, schedule_data).load_features().reset_index().set_index(idx)
        team_feats = TeamFeatures(self.season, odds_sch_matched, self.args.team_halflives).load_features().reset_index().set_index(idx)


        final_features = odds_sch_matched.join(
            [position_player_feats, pitching_feats, context_feats, team_feats],

        )

        assert len(final_features) == len(odds_sch_matched)

        final_features = self._apply_league_average_deltas(
            final_features,
            raw_batting_data=self.cache.get("raw_batting_data", pd.DataFrame()),
            raw_pitching_data=self.cache.get("raw_pitching_data", pd.DataFrame()),
        )

        self.logger.info(f" Adding matchup columns...")
        starter_ewm_cols = ['season'] + [f'ewm_h{hl}' for hl in list(self.args.starter_halflives)]
        starter_cols = ['starter_era', 'starter_babip', 'starter_hard_hit', 'starter_k_percent', 
                        'starter_barrel_percent', 'starter_fip', 'starter_siera', 'starter_stuff',
                        'starter_ev', 'starter_hr_fb', 'starter_wpa']

        starter_matchups = BaseFeatures._add_matchup_cols_diff_same_base(
                                                            df=final_features,
                                                            cols=starter_cols,
                                                            ewm_cols=starter_ewm_cols)
                                                                            
        final_features = final_features.assign(**starter_matchups)
        reliever_ewm_cols = ['season'] + [f'ewm_h{hl}' for hl in list(self.args.reliever_halflives)]
        pen_cols = ['pen_era', 'pen_babip', 'pen_hard_hit', 'pen_k_percent', 
                    'pen_barrel_percent', 'pen_fip', 'pen_siera', 'pen_stuff',
                    'pen_ev', 'pen_hr_fb', 'pen_wpa_li']
        
        
        pen_matchups = BaseFeatures._add_matchup_cols_diff_same_base(
                                                            df=final_features,
                                                            cols=pen_cols,
                                                            ewm_cols=reliever_ewm_cols)
                                                                            
        final_features = final_features.assign(**pen_matchups)

        team_pitch_cols = ["starter_fip", "starter_k_percent", "starter_bb_percent", "starter_barrel_percent"]

        opp_team_bat_cols = ["woba", "k_percent", "bb_percent", "barrel_percent"]
        bat_ewm_cols = ['season'] + [f'ewm_h{hl}' for hl in list(self.args.batter_halflives)]
        team_pitching_vs_opp_batting = BaseFeatures._add_matchup_cols_diff_base(
                                                                        df=final_features,
                                                                        col1=team_pitch_cols,
                                                                        col2=opp_team_bat_cols,
                                                                        col1_ewm_cols=starter_ewm_cols,
                                                                        col2_ewm_cols=bat_ewm_cols)   
        
        final_features = final_features.assign(**team_pitching_vs_opp_batting)

        team_metrics_cols = ["win_pct", "pyth_expectation", "run_diff", "one_run_win_pct"]
        team_metrics_ewm_cols = ['season'] + [f'ewm_h{hl}' for hl in list(self.args.team_halflives)]
        team_metrics_matchups = BaseFeatures._add_matchup_cols_diff_same_base(df=final_features,
                                                                                 cols=team_metrics_cols,
                                                                                 ewm_cols=team_metrics_ewm_cols)
        
        final_features = final_features.assign(**team_metrics_matchups)

        assert (final_features['home_away_starter_fip_woba_season_diff'] == final_features['home_starter_fip_season'] - final_features['away_woba_season']).all()


        final_features = final_features.drop(columns=[col for col in final_features.columns if col.endswith('_sch_bat') or
                                                    col in ['wind', 'condition'] or 
                                                    col.endswith('_to_drop')])

        assert final_features.index.get_level_values('game_date').is_monotonic_increasing, "final_features game_date not globally sorted"

        self.logger.info(f" Final merged dataset shape: {final_features.shape}")
        self.logger.debug(f" Final dataframe datatypes:\n{final_features.dtypes.to_dict()}")
        self.logger.debug(f" Final columns: {final_features.columns.to_list()}")        

        nan_rows = final_features[final_features.isna().any(axis=1)]
        self.logger.info(f" Remaining rows with NaN for {self.season}: {len(nan_rows)}")
        self.logger.debug(f" Sample NaN rows\n{nan_rows.head().to_string()}")

        self.logger.debug("="*60 + "\n")
        self.logger.debug(" Final features DataFrame tail")
        self.logger.debug(final_features.tail().to_string())
        self.logger.debug("="*60 + "\n")

        return final_features, raw_odds_data

        ### TODO
        # if mkt_only:
        #     home_only = transformed_schedule[transformed_schedule['is_home'] == True]
        #     assert len(home_only) * 2 == len(transformed_schedule)

        #     cols = ['day_night_game', 'venue_name', 'venue_id', 'venue_elevation', 'venue_timezone', 'venue_gametime_offset', 'status', 'away_starter_normalized',
        #             'home_starter_normalized', 'wind', 'condition', 'temp', 'winning_team', 'losing_team', 'is_home', 'starter_normalized', 'opposing_starter_normalized',
        #              'team_score' , 'opposing_team_score']

        #     home_only = home_only.drop(columns=cols).reset_index()
        #     home_only.rename(columns={'team': 'home_team', 'opposing_team': 'away_team', 'is_winner': 'is_winner_home'}, inplace=True)
        #     home_only.set_index(['game_date', 'dh', 'game_datetime', 'home_team', 'away_team', 'game_id', 'season'], inplace=True)

        #     return home_only, raw_odds_data
        
        # batting_features = self._get_batting_features(transformed_schedule, force_recreate)
       

        # context_features = GameContextFeatures(self.season, schedule_data).load_features()
        # context_features = context_features.drop(columns=['away_team', 'home_team'])
        

        # team_features = TeamFeatures(self.season, transformed_schedule).load_features()
        # team_feat_cols = ['win_pct_season', 'win_pct_ewm_h3', 'win_pct_ewm_h8', 'win_pct_ewm_h20', 
        #                   'pyth_expectation_season', 'pyth_expectation_ewm_h3', 'pyth_expectation_ewm_h8', 'pyth_expectation_ewm_h20', 
        #                   'run_diff_season', 'run_diff_ewm_h3', 'run_diff_ewm_h8', 'run_diff_ewm_h20',
        #                   'one_run_win_pct_season', 'one_run_win_pct_ewm_h3', 'one_run_win_pct_ewm_h8', 'one_run_win_pct_ewm_h20']
        
        # team_features.rename(columns = {col: f"team_{col}" for col in team_feat_cols}, inplace=True)
        # team_features_with_opp = self._add_opponent_features(team_features, feature_cols=[f"team_{col}" for col in team_feat_cols])
        
        # team_features_with_opp = team_features_with_opp.sort_index(level=['game_date', 'dh', 'team'])
        # team_features_with_opp.rename(columns = {col: f"team_{col}" for col in team_feat_cols}, inplace=True)

        # raw_pitching_data = self._load_pitching_data()
        # pitching_features = PitchingFeatures(self.season, raw_pitching_data, force_recreate).load_features().reset_index()
        # pitching_features = pitching_features.set_index(['game_date', 'dh', 'team', 'opposing_team'])

        # self.logger.info(f" Adding opposing team bullpen pitching stats to each row for {self.season}")
        # bullpen_cols = [col for col in pitching_features.columns if 'pen_' in col and not col.startswith('opposing_')]

        # pitching_features_opp_bp = self._add_opponent_features(pitching_features, feature_cols=bullpen_cols)
        # pitching_features_opp_bp = pitching_features_opp_bp.sort_index(level=['game_date', 'dh', 'team'])
        # pitching_features_opp_bp = pitching_features_opp_bp.reset_index()

        # transformed_schedule_reset = transformed_schedule.reset_index()
        # batting_features_reset = batting_features.reset_index()
        
        # sch_batting_features = pd.merge(
        #     transformed_schedule_reset,
        #     batting_features_reset,
        #     on=['game_date', 'game_id', 'dh', 'team', 'opposing_team'],
        #     how='inner',
        #     validate='1:1' 
        # )

        # sch_bat_ctx = pd.merge(
        #     sch_batting_features,
        #     context_features,
        #     on=['game_id', 'game_date', 'dh', 'game_datetime'],
        #     how='inner',
        #     validate='m:1',
        #     suffixes=('_sch_bat', '')
        # )
        
        # sch_bat_ctx_team = pd.merge(
        #     sch_bat_ctx,
        #     team_features_with_opp,
        #     on=['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team'],
        #     how='inner',
        #     validate='1:1',
        # )

        # final_features = pd.merge(
        #     sch_bat_ctx_team,
        #     pitching_features_opp_bp,
        #     on=['game_date', 'dh', 'team', 'opposing_team', 'season'],
        #     how='inner',
        #     validate='1:1',
        #     suffixes=('_to_drop', '_to_drop')
        # )


        # return final_features, raw_odds_data
    
    def _merge_batting_fielding_features(self, batting_stats: DataFrame, fielding_stats: DataFrame) -> DataFrame:
        bat_df = batting_stats.copy()
        fld_df = fielding_stats.copy()

        bat_df.sort_values(['mlb_id', 'game_date', 'dh'], inplace=True)
        fld_df.sort_values(['player_id', 'month'], inplace=True)

        bat_df['month'] = bat_df['game_date'].dt.month
        
        bat_df['month'] = bat_df['month'].map({
            3: 4,
            10: 9
        }).fillna(bat_df['month'])

        merged_df = pd.merge(
            bat_df,
            fld_df,
            left_on=['mlb_id', 'month'],
            right_on=['player_id', 'month'],
            how='left',
            suffixes=("", "_fld")
        )

        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_fld')])

        return merged_df
    
    def _merge_schedule_with_position_player_features(self, schedule_df: DataFrame, lineups_data: DataFrame, batting_features: DataFrame) -> DataFrame:
        """
        Merges schedule, lineups, and rolling position player features into one DataFrame.
        """
        games = (
            schedule_df.reset_index()[["game_id", "game_date", "dh", "home_team", "away_team"]]
            .drop_duplicates()
        )

        cols = ["player_id", "position"]
        away_renamed_cols = ["away_" + col for col in cols]
        home_renamed_cols = ["home_" + col for col in cols]

        away_lineups_for_games = games.merge(
            lineups_data,
            left_on=["game_date", "dh", "away_team", "home_team"],
            right_on=["game_date", "dh", "team", "opposing_team"],
            how="left",
        )
        away_lineups_for_games = away_lineups_for_games.rename(columns=dict(zip(cols, away_renamed_cols)))

        home_lineups_for_games = games.merge(
            lineups_data,
            left_on=["game_date", "dh", "home_team", "away_team"],
            right_on=["game_date", "dh", "team", "opposing_team"],
            how="left",
        )
        home_lineups_for_games = home_lineups_for_games.rename(columns=dict(zip(cols, home_renamed_cols)))
        batting_features['player_id'] = batting_features['player_id'].astype('int')
        drop_cols = ["team", "opposing_team", "player_id", "team_bf", "season_bf"]

        ewm_suffixes = [f'_ewm_h{hl}' for hl in self.args.batter_halflives]
        stats_cols = [
            c for c in batting_features.columns
            if any(s in c for s in ['_season', *ewm_suffixes, 'frv_per_9'])
        ]
        league_medians = batting_features[stats_cols].median(numeric_only=True)
        group_cols = ["game_id", "game_date", "home_team", "away_team", "dh"]

        away_lineups_with_stats = away_lineups_for_games.merge(
            batting_features,
            left_on=["game_date", "dh", "away_player_id"],
            right_on=["game_date", "dh", "player_id"],
            how="left",
            suffixes=("", "_bf"),
        )
        away_lineups_with_stats = away_lineups_with_stats.rename(columns={"mlb_id": "away_mlb_id"})
        away_lineups_with_stats = away_lineups_with_stats.drop(columns=drop_cols, errors="ignore")
        away_lineups_with_stats[stats_cols] = away_lineups_with_stats[stats_cols].fillna(league_medians)
        away_position_player_features = (
            away_lineups_with_stats
            .groupby(group_cols, observed=True)[stats_cols]
            .mean()
            .rename(columns={col: f"away_{col}" for col in stats_cols})
        )
        
        home_lineups_with_stats = home_lineups_for_games.merge(
            batting_features,
            left_on=["game_date", "dh", "home_player_id"],
            right_on=["game_date", "dh", "player_id"],
            how="left",
            suffixes=("", "_bf"),
        )
        home_lineups_with_stats = home_lineups_with_stats.rename(columns={"mlb_id": "mlb_id_home"})
        home_lineups_with_stats = home_lineups_with_stats.drop(columns=drop_cols, errors="ignore")
        home_lineups_with_stats[stats_cols] = home_lineups_with_stats[stats_cols].fillna(league_medians)
        home_position_player_features = (
            home_lineups_with_stats
            .groupby(group_cols, observed=True)[stats_cols]
            .mean()
            .rename(columns={col: f"home_{col}" for col in stats_cols})
        )

        return away_position_player_features.join(home_position_player_features, how="outer")
    
    def _get_position_player_features(self, schedule_df: DataFrame, force_recreate: bool = False) -> DataFrame:
        """Get position_player features for all games efficiently"""
        raw_batter_data = self._load_batting_data()
        self.cache["raw_batting_data"] = raw_batter_data.copy()
        raw_batter_data = self._filter_position_player_batting_data(raw_batter_data)
        previous_batter_data = self._filter_position_player_batting_data(self._load_previous_batting_data())

        lineups_data = self._load_lineups_data()
        lineups_data['game_date'] = pd.to_datetime(lineups_data['game_date'])
        lineups_data = lineups_data[~(lineups_data['position'].str.startswith('P') & (lineups_data['player_id'] != 19755))]

        batting_features = BattingFeatures(
            self.season,
            raw_batter_data,
            force_recreate,
            self.args.batter_halflives,
            previous_season_data=previous_batter_data,
        )
        
        self.logger.info(f" Calculating batting rolling stats for {self.season}")
        batting_features = batting_features.load_features()

        raw_fielding_data = self._load_fielding_data()
        fielding_feats = FieldingFeatures(self.season, raw_fielding_data).load_features()

        self.logger.info(f" Merging batting rolling and fielding stats for {self.season}")
        bat_fld_feats = self._merge_batting_fielding_features(batting_features, fielding_feats)
        bat_fld_feats["frv_per_9"] = bat_fld_feats["frv_per_9"].fillna(0.0)

        self.logger.info(f" Merging schedule, lineups, and position player rolling stats for {self.season}")
        team_position_player_features = self._merge_schedule_with_position_player_features(schedule_df, lineups_data, bat_fld_feats)

        self.logger.info(f" Adding team batting comparison stats for {self.season}")
        batting_comp_cols = ["woba", "wrc_plus", "hard_hit", "barrel_percent", "bb_k", "ops",
                             "babip", "ev", "iso", "baserunning", "wpa"]
        ewm_cols = ['season'] + [f'ewm_h{hl}' for hl in self.args.batter_halflives]

        batting_comp_stats = BaseFeatures._add_matchup_cols_diff_same_base(team_position_player_features, batting_comp_cols, ewm_cols)
        team_position_player_features = team_position_player_features.assign(**batting_comp_stats)

        return team_position_player_features

    def _get_pitcher_features(self, schedule_df: DataFrame, force_recreate: bool = False) -> DataFrame:
        """
        Merges schedule and all pitching features into one DataFrame. 
        """
        games = (
            schedule_df.reset_index()[["game_id", "game_date", "dh", "home_team", "away_team"]]
            .drop_duplicates()
        )

        raw_pitching_data = self._load_pitching_data()
        self.cache["raw_pitching_data"] = raw_pitching_data.copy()
        raw_feats = PitchingFeatures(
            self.season, 
            raw_pitching_data, 
            force_recreate, 
            self.args.starter_halflives, 
            self.args.reliever_halflives,
            previous_season_data=self._load_previous_pitching_data(),
        ).load_features()
        
        team_starter_cols = [col for col in raw_feats.columns if col.startswith("team_starter")]
        pen_cols = [col for col in raw_feats.columns if col.startswith("team_pen")]
        pitching_keys = ["game_date", "dh", "team", "opposing_team"]

        home_stats = games.merge(
            raw_feats[pitching_keys + team_starter_cols + pen_cols],
            left_on=["game_date", "dh", "home_team", "away_team"],
            right_on=["game_date", "dh", "team", "opposing_team"],
            how="left"
        )

        home_rename_cols = [col.replace("team", "home") for col in team_starter_cols + pen_cols]
        home_stats = home_stats.drop(columns=["team", "opposing_team", "team_starter_player_id"])
        home_stats = home_stats.rename(columns=dict(
            zip(team_starter_cols + pen_cols, home_rename_cols)
            )).set_index(["game_id", "game_date", "dh", "home_team", "away_team"])

        away_stats = games.merge(
            raw_feats[pitching_keys + team_starter_cols + pen_cols],
            left_on=["game_date", "dh", "away_team", "home_team"],
            right_on=["game_date", "dh", "team", "opposing_team"],
            how="left"
        )

        away_rename_cols = [col.replace("team", "away") for col in team_starter_cols + pen_cols]
        away_stats = away_stats.drop(columns=["team", "opposing_team", "team_starter_player_id"])
        away_stats = away_stats.rename(columns=dict(
            zip(team_starter_cols + pen_cols, away_rename_cols)
            )).set_index(["game_id", "game_date", "dh", "home_team", "away_team"])

        all_pitching_features = pd.merge(
            home_stats,
            away_stats,
            left_index=True,
            right_index=True,
            how="inner"
        )

        assert len(all_pitching_features) == len(games)
        
        return all_pitching_features

    def _apply_league_average_deltas(
        self,
        features: DataFrame,
        raw_batting_data: DataFrame,
        raw_pitching_data: DataFrame,
    ) -> DataFrame:
        """
        Convert supported home/away rolling stats to deltas from dynamic league averages.

        Positive values mean better than league average. League averages come from raw
        season rows before the feature row date, with a full-season fallback early in
        the season when fewer than 10 prior league buckets are available.
        """
        adjusted = features.copy()
        if adjusted.empty:
            return adjusted

        feature_dates = self._feature_dates(adjusted)

        batting_stats = {
            "woba": ("pa", "higher"),
            "ops": ("pa", "higher"),
            "bb_percent": ("pa", "higher"),
            "bb_k": ("pa", "higher"),
            "babip": ("bip", "higher"),
            "barrel_percent": ("bip", "higher"),
            "hard_hit": ("bip", "higher"),
            "ev": ("bip", "higher"),
            "gb_fb": ("bip", "higher"),
            "iso": ("ab", "higher"),
            "k_percent": ("pa", "lower"),
        }
        pitching_stats = {
            "era": ("ip", "lower"),
            "fip": ("ip", "lower"),
            "siera": ("ip", "lower"),
            "k_percent": ("tbf", "higher"),
            "bb_percent": ("tbf", "lower"),
            "k_bb_percent": ("tbf", "higher"),
            "babip": ("bip", "lower"),
            "barrel_percent": ("bip", "lower"),
            "hard_hit": ("bip", "lower"),
            "ev": ("bip", "lower"),
            "hr_fb": ("bip", "lower"),
        }

        raw_batting = self._filter_league_average_batting(raw_batting_data)
        batting_averages = {
            stat: self._league_averages_by_feature_date(raw_batting, stat, denom, feature_dates)
            for stat, (denom, _) in batting_stats.items()
        }
        for stat, (_, direction) in batting_stats.items():
            adjusted = self._apply_stat_delta(adjusted, stat, direction, batting_averages[stat])

        raw_pitching = self._prepare_pitching_league_average_data(raw_pitching_data)
        starter_rows, reliever_rows = self._split_pitching_rows(raw_pitching)
        starter_averages = {
            stat: self._league_averages_by_feature_date(starter_rows, stat, denom, feature_dates)
            for stat, (denom, _) in pitching_stats.items()
        }
        reliever_averages = {
            stat: self._league_averages_by_feature_date(reliever_rows, stat, denom, feature_dates)
            for stat, (denom, _) in pitching_stats.items()
        }

        for stat, (_, direction) in pitching_stats.items():
            adjusted = self._apply_stat_delta(adjusted, f"starter_{stat}", direction, starter_averages[stat])
            adjusted = self._apply_stat_delta(adjusted, f"pen_{stat}", direction, reliever_averages[stat])

        return adjusted

    def _feature_dates(self, features: DataFrame) -> pd.Series:
        if "game_date" in features.index.names:
            dates = features.index.get_level_values("game_date")
        elif "game_date" in features.columns:
            dates = features["game_date"]
        else:
            raise ValueError("features must include game_date")
        return pd.Series(pd.to_datetime(dates), index=features.index)

    def _filter_league_average_batting(self, raw_batting_data: DataFrame) -> DataFrame:
        if raw_batting_data is None or raw_batting_data.empty:
            return pd.DataFrame()

        raw = raw_batting_data.copy()
        is_pitcher = raw["pos"].fillna("").astype(str).str.startswith("P")
        raw = raw[~(is_pitcher & (raw["player_id"] != 19755))]
        return raw

    def _prepare_pitching_league_average_data(self, raw_pitching_data: DataFrame) -> DataFrame:
        if raw_pitching_data is None or raw_pitching_data.empty:
            return pd.DataFrame()

        raw = raw_pitching_data.copy()
        if "k_bb_percent" not in raw.columns and {"k_percent", "bb_percent"}.issubset(raw.columns):
            raw["k_bb_percent"] = raw["k_percent"] - raw["bb_percent"]
        return raw

    def _split_pitching_rows(self, raw_pitching_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if raw_pitching_data is None or raw_pitching_data.empty:
            empty = pd.DataFrame()
            return empty, empty

        raw = raw_pitching_data.copy()
        if "is_starter" in raw.columns:
            is_starter = raw["is_starter"].fillna(False).astype(bool)
        elif "gs" in raw.columns:
            gs = pd.to_numeric(raw["gs"], errors="coerce")
            is_starter = gs > 0
        else:
            return raw, raw

        return raw[is_starter], raw[~is_starter]

    def _league_averages_by_feature_date(
        self,
        raw_data: DataFrame,
        stat: str,
        denominator: str,
        feature_dates: pd.Series,
    ) -> pd.Series:
        if raw_data is None or raw_data.empty or stat not in raw_data.columns:
            return pd.Series(np.nan, index=feature_dates.index)

        raw = raw_data.copy()
        raw["game_date"] = pd.to_datetime(raw["game_date"])
        raw[stat] = pd.to_numeric(raw[stat], errors="coerce")

        if denominator in raw.columns:
            raw["_denominator"] = pd.to_numeric(raw[denominator], errors="coerce")
        else:
            raw["_denominator"] = 1.0

        raw = raw.dropna(subset=["game_date", stat, "_denominator"])
        raw = raw[raw["_denominator"] > 0]
        if raw.empty:
            return pd.Series(np.nan, index=feature_dates.index)

        full_average = self._weighted_average(raw, stat)
        averages_by_date = {}
        for game_date in feature_dates.drop_duplicates().sort_values():
            prior = raw[raw["game_date"] < game_date]
            if self._prior_league_bucket_count(prior) < 10:
                averages_by_date[game_date] = full_average
            else:
                averages_by_date[game_date] = self._weighted_average(prior, stat)

        return feature_dates.map(averages_by_date)

    def _prior_league_bucket_count(self, prior_raw_data: DataFrame) -> int:
        if prior_raw_data.empty:
            return 0

        if {"game_date", "dh", "team"}.issubset(prior_raw_data.columns):
            bucket_cols = ["game_date", "dh", "team"]
        elif {"game_date", "team"}.issubset(prior_raw_data.columns):
            bucket_cols = ["game_date", "team"]
        else:
            bucket_cols = ["game_date"]

        return len(prior_raw_data[bucket_cols].drop_duplicates())

    def _weighted_average(self, raw_data: DataFrame, stat: str) -> float:
        denominator = raw_data["_denominator"].sum()
        if denominator == 0:
            return np.nan
        return (raw_data[stat] * raw_data["_denominator"]).sum() / denominator

    def _apply_stat_delta(
        self,
        features: DataFrame,
        stat: str,
        direction: str,
        averages: pd.Series,
    ) -> DataFrame:
        if averages.isna().all():
            return features

        adjusted = features
        prefixes = (f"home_{stat}_", f"away_{stat}_")
        columns = [col for col in adjusted.columns if col.startswith(prefixes)]
        for col in columns:
            if direction == "lower":
                adjusted[col] = averages - adjusted[col]
            else:
                adjusted[col] = adjusted[col] - averages
        return adjusted




    def _add_opponent_features(self, df:DataFrame,
                               team_level='team',
                               opp_level='opposing_team',
                               feature_cols=None) -> DataFrame:
        """
        df has a MultiIndex that includes team and opposing_team (plus game keys).
        Returns df with opponent's columns prefixed 'opposing_'.
        """
        if feature_cols is None:
            feature_cols = df.columns

        feature_cols = [col for col in feature_cols if not col.startswith('opposing')]

        if df.index.has_duplicates:
            raise ValueError("Input index has duplicates; disambiguate (e.g., include dh/game_datetime).")

        opp_view = df[feature_cols].copy()
        opp_view.index = opp_view.index.swaplevel(team_level, opp_level)
        opp_view = opp_view.sort_index()

        opp_aligned = opp_view.reindex(df.index)
        opp_aligned.columns = [f"opposing_{c}" for c in feature_cols]

        return pd.concat([df, opp_aligned], axis=1)    

    def _match_schedule_to_odds(self, schedule_data: DataFrame, odds_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Match schedule games to odds. A single game will have many odds matches. Games without a match are due to incorrect labeling
        of game_datetime in the schedule data, so the odds game_datetime is used. This is handled in find_unmatched_games and _handle_unmatched_games
        """

        def find_unmatched_games(merged_games: DataFrame, odds_data: DataFrame, keys: List[str] = []) -> DataFrame:
            if not keys:
                keys = ['game_date', 'game_datetime', 'away_team', 'home_team'] 
            
            unmatched_games = odds_data.merge(
                merged_games[keys], how='left', on=keys, indicator=True
            )

            unmatched_games = unmatched_games[unmatched_games['_merge'] == 'left_only'].drop(columns=['_merge'])
            self.logger.info(f" Unmatched games: {len(unmatched_games)}")

            return unmatched_games
        
        def check_column_mismatches(merged_df: DataFrame, merge_type: str = ""):
            """Check for mismatches between _sch and _odds columns and log/save details"""
            self.logger.debug(" Checking for column missmatches...")
            for col in merged_df.columns:
                if col.endswith('_sch'):
                    base_name = col[:-4] 
                    odds_col = f"{base_name}_odds"

                    if odds_col in merged_df.columns:
                        mismatch = merged_df[col] != merged_df[odds_col]

                        if mismatch.any():
                            self.logger.warning(f" WARNING{merge_type}: {base_name} has {mismatch.sum()} mismatches")
                           
                            mismatched_rows = merged_df[mismatch]
                            mismatch_file = f"{base_name}_mismatches{merge_type.lower().replace(' ', '_')}.txt"

                            self.logger.debug(f" Mismatches for column: {base_name}{merge_type}\n")
                            self.logger.debug("="*50 + "\n")

                            for _, row in mismatched_rows.iterrows():
                                sch_val = row[col]
                                odds_val = row[odds_col]
                                game_info = f"{row.get('game_date', 'N/A')} {row.get('dh', 'N/A')} {row.get('away_team', 'N/A')} @ {row.get('home_team', 'N/A')}"
                                self.logger.debug(f"Game: {game_info}\n")
                                self.logger.debug(f"  Schedule value: {sch_val}\n")
                                self.logger.debug(f"  Odds value: {odds_val}\n")
                                self.logger.debug("-" * 30 + "\n")

                            self.logger.info(f" Detailed mismatches saved to {mismatch_file}")
        
        keys_ = ['game_date', 'game_datetime', 'away_team', 'home_team']

        merged_games = pd.merge(
            schedule_data,
            odds_data,
            how='inner',
            on=keys_,
            suffixes=('', '_rm')
        )
        merged_games = merged_games.drop(columns=[col for col in merged_games.columns if col.endswith('_rm')])

        
        self.logger.info(f" After basic merge: {len(merged_games)} rows")
        self.logger.info(f" Unique games in schedule: {schedule_data[['game_date', 'dh', 'away_team', 'home_team']].drop_duplicates().shape[0]}")
        self.logger.info(f" Unique games in odds: {odds_data[['game_date', 'game_datetime', 'away_team', 'home_team']].drop_duplicates().shape[0]}")

        check_column_mismatches(merged_games, " (Initial Merge)")

        unmatched_games = find_unmatched_games(merged_games=merged_games, odds_data=odds_data)
   
        if not unmatched_games.empty:
            updated_odds_with_dh = self._handle_unmatched_games(schedule_data, unmatched_games)

            if not updated_odds_with_dh.empty:
                updated_merged_games = pd.merge(
                    schedule_data,
                    updated_odds_with_dh,
                    how='inner',
                    on=['game_date', 'away_team', 'home_team', 'dh'],
                    suffixes=('', '_odds')
                )

                check_column_mismatches(updated_merged_games, " (Unmatched Games Merge)")

                cols_to_drop = [col for col in updated_merged_games.columns 
                               if col.endswith('_odds') and not col in ['away_opening_odds', 'home_opening_odds']]
                
                updated_merged_games = updated_merged_games.drop(columns=cols_to_drop)

                all_merged_keys = pd.concat([
                    merged_games[['game_date', 'away_team', 'home_team']],
                    updated_merged_games[['game_date', 'away_team', 'home_team']]
                ]).drop_duplicates()
                

                remaining_unmatched_games = find_unmatched_games(
                    merged_games=all_merged_keys, 
                    odds_data=odds_data, 
                    keys=['game_date', 'away_team', 'home_team']
                )

                self.logger.info(f" Remaining unmatched games: {len(remaining_unmatched_games)}")
                
                all_odds_data = pd.concat([merged_games, updated_odds_with_dh], ignore_index=True)
                all_odds_data = all_odds_data.drop_duplicates(
                    subset=['game_date', 'away_team', 'home_team', 'sportsbook']
                )
                
                final_merged = pd.merge(
                    schedule_data,
                    all_odds_data,
                    how='left',
                    on=['game_date', 'away_team', 'home_team', 'dh'],
                    suffixes=('', '_to_drop')
                )

                final_merged = final_merged.drop_duplicates(
                    subset=['game_date', 'game_id', 'away_team', 'home_team', 'dh']
                )
                
                cols_to_drop = [col for col in final_merged.columns 
                               if col.endswith('_to_drop') or 
                               col in ['away_starter', 'home_starter', 'winner'] and 
                               not col in ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']]
                
                final_merged = final_merged.drop(columns=cols_to_drop)
                final_merged = final_merged.set_index(['game_date', 'dh', 'game_datetime', 'away_team', 'home_team'])
                self.logger.info(f" Schedule games after left merge: {len(final_merged)}")
                self.logger.info(f" Odds rows still unmatched to schedule: {len(remaining_unmatched_games)}")
                
                original_odds_count = len(odds_data)
                all_odds_count = len(all_odds_data)
                self.logger.info(f" Original odds: {original_odds_count}, All merged odds (including reconciled): {all_odds_count}")
                
                if all_odds_count > original_odds_count:
                    self.logger.warning(f" Merged games ({all_odds_count}) exceed total available odds ({original_odds_count})!")
                else:
                    self.logger.info(f" Merge validation passed: {all_odds_count} merged <= {original_odds_count} total odds")

                self._log_matching_summary(schedule_data, odds_data, final_merged, remaining_unmatched_games)
                merged_games = final_merged
                all_odds_data.set_index(["game_date", "dh", "home_team", "away_team", "game_id"], inplace=True)
                all_odds_data.drop(columns=[col for col in all_odds_data.columns if col not in ['sportsbook', 'away_opening_prob_raw', 'home_opening_prob_raw', 'home_opening_prob_nv', 'away_opening_prob_nv']], inplace=True)
            else:
                self.logger.warning("handle_unmatched_games returned no matches")

        ## REFACTOR LATER
        id_cols = ["game_date", "dh", "game_datetime", 'away_team', 'home_team']

        merged_games = merged_games.reset_index()
        merged_games = merged_games.sort_values(id_cols).set_index(id_cols)
        
        game_metadata_cols = [
            'game_id', 'season', 'venue_timezone', 'venue_gametime_offset', 'status',
            'away_probable_pitcher', 'home_probable_pitcher', 'away_starter_normalized',
            'home_starter_normalized', 'wind', 'condition', 'away_score',
            'home_score', 'winning_team', 'losing_team', 'away_starter', 'home_starter',
            'winner'
        ]
        
        aggregated_odds_cols = [
            'vig_open', 'p_open_home_median', 'p_open_home_mean', 'p_open_home_std', 
            'p_open_away_median', 'p_open_away_mean', 'p_open_away_std',  
            'p_open_home_max_nv', 'p_open_home_min_nv', 'p_open_away_max_nv',
            'p_open_away_min_nv', 'p_open_home_median_nv', 'p_open_home_mean_nv', 
            'p_open_home_std_nv', 'p_open_away_median_nv', 'p_open_away_mean_nv', 
            'p_open_away_std_nv', 'p_open_mean_nv_diff', 'p_open_med_nv_diff', 
            'p_open_max_nv_diff', 'p_open_min_nv_diff', 'num_books', 'logit_prob_home_std_nv'
        ]
        
        existing_metadata_cols = [col for col in game_metadata_cols if col in merged_games.columns]
        existing_odds_cols = [col for col in aggregated_odds_cols if col in merged_games.columns]
        
        metadata_df = merged_games.groupby(level=id_cols)[existing_metadata_cols].first()
        
        if existing_odds_cols:
            odds_df = merged_games.groupby(level=id_cols)[existing_odds_cols].first()
            final_result = metadata_df.join(odds_df, how='left')
        else:
            final_result = metadata_df
        
        odds_cols = [col for col in final_result.columns if col.startswith('p_')] + ['vig_open']
        no_odds_mask = final_result[odds_cols].isna().any(axis=1)
        no_odds_rows = final_result[no_odds_mask]
        self.logger.info(f" Games with no odds:\n{no_odds_rows}")
        self.logger.info(f" Removing games without odds...")
        final_result = final_result[~no_odds_mask]

        self.logger.debug("="*50 + "\n")
        self.logger.debug(" Resulting DataFrame after matching schedule")       
        self.logger.debug(final_result.head().to_string())
        self.logger.debug("="*50 + "\n")
        
        return final_result, all_odds_data

    def _log_matching_summary(self, schedule_data: DataFrame, odds_data: DataFrame, 
                               final_merged: DataFrame, remaining_unmatched: DataFrame):
        """Print a comprehensive summary of the schedule-to-odds matching process"""
        self.logger.debug("\n" + "="*60)
        self.logger.debug(" SCHEDULE-TO-ODDS MATCHING SUMMARY")
        self.logger.debug("="*60)
        
        schedule_games = schedule_data[['game_date', 'away_team', 'home_team']].drop_duplicates()
        odds_games = odds_data[['game_date', 'away_team', 'home_team']].drop_duplicates()
        merged_games = final_merged.reset_index()[['game_date', 'away_team', 'home_team']].drop_duplicates()
        
        self.logger.debug(f" Total unique games in schedule: {len(schedule_games)}")
        self.logger.debug(f" Total unique games in odds: {len(odds_games)}")
        self.logger.debug(f" Successfully matched unique games: {len(merged_games)}")
        self.logger.debug(f" Remaining unmatched odds entries: {len(remaining_unmatched)}")
        
        match_rate = (len(merged_games) / len(odds_games)) * 100 if len(odds_games) > 0 else 0
        self.logger.debug(f" Match rate: {match_rate:.1f}%")
        
        if len(remaining_unmatched) > 0:
            self.logger.debug(f"\nSample unmatched games (first 5):")
            sample_cols = ['game_date', 'game_datetime', 'away_team', 'home_team']
            self.logger.debug(remaining_unmatched[sample_cols].head().to_string(index=False))
        
        self.logger.debug("="*60 + "\n")


    def _handle_unmatched_games(self, schedule: DataFrame, unmatched_games: DataFrame) -> DataFrame:
        """
        Reconciles unmatched games from schedule to odds using smart datetime matching.
        For doubleheaders, matches odds to the closest game time to prevent duplication.
        """
            
        if unmatched_games.empty:
            return DataFrame()
        
        self.logger.info(" Reconciling unmatched games with datetime matching by time distance")
        schedule = schedule.reset_index().copy()
        
        schedule['game_datetime'] = pd.to_datetime(schedule['game_datetime'])
        unmatched_games['game_datetime'] = pd.to_datetime(unmatched_games['game_datetime'])
        
        reconciled_games = []
        
        for (date, away, home), group in unmatched_games.groupby(['game_date', 'away_team', 'home_team']):

            schedule_games = schedule[
                (schedule['game_date'] == date) & 
                (schedule['away_team'] == away) & 
                (schedule['home_team'] == home)
            ].copy()
            
            if len(schedule_games) == 0:
                self.logger.warning(f" No schedule match for {date} {away} @ {home}")
                continue
            elif len(schedule_games) == 1:
                for _, odds_row in group.iterrows():
                    merged_row = odds_row.copy()
                    merged_row['game_id'] = schedule_games.iloc[0]['game_id']
                    merged_row['dh'] = schedule_games.iloc[0]['dh']
                    reconciled_games.append(merged_row)
            else:
                schedule_games = schedule_games.sort_values('dh')
                
                for _, odds_row in group.iterrows():
                    odds_time = odds_row['game_datetime']
                    
                    time_diffs = []
                    for _, sch_game in schedule_games.iterrows():
                        sch_time = sch_game['game_datetime']
                        time_diff = abs((odds_time - sch_time).total_seconds())
                        time_diffs.append((time_diff, sch_game))
                    
                    closest_game = min(time_diffs, key=lambda x: x[0])[1]
                    
                    merged_row = odds_row.copy()
                    merged_row['game_id'] = closest_game['game_id']
                    merged_row['dh'] = closest_game['dh']
                    reconciled_games.append(merged_row)
                    
                    self.logger.debug(f"Matched {odds_row['sportsbook']} odds for {date} {away}@{home} "
                               f"(odds time: {odds_time}) to Game {int(closest_game['dh'])} "
                               f"(schedule time: {closest_game['game_datetime']})")
        
        if reconciled_games:
            result_df = DataFrame(reconciled_games)

            result_df = result_df.drop(columns=['game_datetime'], errors='ignore')
            
            self.logger.info(f" Game reconciliation completed: {len(result_df)} odds matched")
            
            unique_matches = result_df.groupby(['game_date', 'away_team', 'home_team', 'sportsbook', 'dh']).size()
            if (unique_matches > 1).any():
                self.logger.warning("Duplicate matches detected in reconciliation!")
            
            return result_df
        else:
            return DataFrame()
        
    def _load_schedule_data(self) -> DataFrame:
        game_loader = GameLoader()
        schedule_data = game_loader.load_for_season(self.season)
        return schedule_data
    
    def _load_odds_data(self) -> DataFrame:
        odds_loader = OddsLoader()
        odds_data = odds_loader.load_for_season(self.season)
        return odds_data
    
    def _load_batting_data(self) -> DataFrame:
        loader = PlayerLoader()
        batting_data = loader.load_for_season_batter(self.season)
        if batting_data.empty:
            raise ValueError(f"Batting data in {self.season} is empty")
        return batting_data

    def _load_previous_batting_data(self) -> DataFrame:
        loader = PlayerLoader()
        return loader.load_for_season_batter(self.season - 1)

    def _filter_position_player_batting_data(self, batting_data: DataFrame) -> DataFrame:
        if batting_data is None or batting_data.empty or "pos" not in batting_data.columns:
            return batting_data

        is_pitcher = batting_data["pos"].fillna("").astype(str).str.startswith("P")
        return batting_data[~(is_pitcher & (batting_data["player_id"] != 19755))].copy()
    
    def _load_lineups_data(self) -> DataFrame:
        loader = TeamLoader()
        lineups_data = loader.load_lineup_players(self.season)
        lineups_data["player_id"] = lineups_data["player_id"].astype(int)
        return lineups_data
    
    def _load_pitching_data(self) -> DataFrame:
        loader = PlayerLoader()
        pitching_data = loader.load_for_season_pitcher(self.season)
        return pitching_data

    def _load_previous_pitching_data(self) -> DataFrame:
        loader = PlayerLoader()
        return loader.load_for_season_pitcher(self.season - 1)
    
    def _load_fielding_data(self) -> DataFrame:
        loader = PlayerLoader()
        fielding_data = loader.load_fielding_stats(self.season)
        return fielding_data
        
def main():
    args = create_args()
    logger = setup_logging("feature_pipeline", LOG_FILE, args=args)
    
    feat_pipe = FeaturePipeline(2021, args, logger)

    features, _ = feat_pipe.start_pipeline(args.force_recreate, mkt_only=True)

if __name__ == "__main__":
    main()
    

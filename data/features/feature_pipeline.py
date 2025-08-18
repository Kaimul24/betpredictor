#!/usr/bin/env python3
"""
Orchestrates the feature engineering process for all features. Applies normalization.
"""

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import logging, sys, argparse
import numpy as np
from typing import List
logger = None
args = None
from src.config import PROJECT_ROOT

from data.features.game_features.context import GameContextFeatures
from data.features.game_features.odds import Odds
from data.features.player_features.batting import BattingFeatures
from data.features.player_features.fielding import FieldingFeatures
from data.features.player_features.pitching import PitchingFeatures
from data.features.player_features.war import WAR
from data.features.team_features.team_features import TeamFeatures

from data.loaders.game_loader import GameLoader
from data.loaders.odds_loader import OddsLoader
from data.loaders.player_loader import PlayerLoader
from data.loaders.team_loader import TeamLoader

LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_pipeline.log"

def create_args():
    """Parse command line arguments"""
    global args
    parser = argparse.ArgumentParser(description="Feature engineering runner")
    parser.add_argument("--force_recreate", action="store_true", help="Recreate batting rolling features, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    args = parser.parse_args()

def setup_logging():
    """Configure logging based on CLI arguments"""
    logger = logging.getLogger("feature_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False 

    if logger.handlers:
        logger.handlers.clear()
    
    fmt = logging.Formatter(
        "%(levelname)s:%(name)s:%(message)s"
    )
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    

    if hasattr(args, 'log') and args.log:
        log_file = args.log_file if hasattr(args, 'log_file') and args.log_file else LOG_FILE
        
        if hasattr(args, 'clear_log') and args.clear_log:
            try:
                with open(log_file, 'w') as f:
                    pass
                logger.info(f"Cleared log file: {log_file}")
            except Exception as e:
                logger.info(f"Warning: Could not clear log file {log_file}: {e}")
        
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

class FeaturePipeline():

    def __init__(self, season: int):
        self.season = season
        self.cache = {}
        self.logger = setup_logging()

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
    
    def _load_lineups_data(self) -> DataFrame:
        loader = TeamLoader()
        lineups_data = loader.load_lineup(self.season)
        return lineups_data
    
    def _transform_schedule(self, schedule_data: DataFrame) -> DataFrame:
        """Splits each game in the schedule into 2 rows, each representing one team's perspective of the game"""
        self.logger.info(f" Transforming schedule for {self.season}")

        schedule_data = schedule_data.reset_index().copy()

        unique_cols = ['away_team', 'home_team', 'away_probable_pitcher', 'home_probable_pitcher', 'starter_normalized', 'opposing_starter_normalized', 'away_score', 'home_team_score']
        common_cols = [col for col in schedule_data.columns if col not in unique_cols]

        away_df_base = schedule_data[common_cols]
        home_df_base = schedule_data[common_cols]

        away_df = away_df_base.assign(
            is_home = False,
            team=schedule_data['away_team'],
            opposing_team=schedule_data['home_team'],
            starter_normalized=schedule_data['away_starter_normalized'],
            opposing_starter_normalized=schedule_data['home_starter_normalized'],
            team_score=schedule_data['away_score'],
            opposing_team_score = schedule_data['home_score'],
            is_winner=np.where(schedule_data['winning_team'] == schedule_data['away_team'], 1, 0)
        )


        home_df = home_df_base.assign(
            is_home = True,
            team=schedule_data['home_team'],
            opposing_team=schedule_data['away_team'], 
            starter_normalized=schedule_data['home_starter_normalized'],
            opposing_starter_normalized=schedule_data['away_starter_normalized'],
            team_score=schedule_data['home_score'],
            opposing_team_score = schedule_data['away_score'],
            is_winner=np.where(schedule_data['winning_team'] == schedule_data['home_team'], 1, 0)
        )

        result = pd.concat([away_df, home_df], ignore_index=True)
        result = result.sort_values(["game_date", "dh", "game_datetime", "is_home"], ascending=[True, True, True, False])
        result = result.set_index(['game_date', 'dh', 'game_datetime', 'team']).sort_index()

        assert len(result) == len(schedule_data) * 2

        return result
    
    def _merge_schedule_with_batting_features(self, schedule_df: DataFrame, lineups_data: DataFrame, batting_features: DataFrame) -> DataFrame:
        """
        Transforms schedule, lineups, and rolling batting features into one DataFrame.
        """
        games = (
            schedule_df.reset_index()[["game_id", "game_date", "dh", "team", "opposing_team"]]
            .drop_duplicates()
        )

        lineups_for_games = games.merge(
            lineups_data[["game_date", "team", "opposing_team", "dh", "player_id", "batting_order", "position"]],
            on=["game_date", "dh", "team", "opposing_team"],
            how="inner",
            validate="1:m"
        )

        lineups_with_stats = lineups_for_games.merge(
            batting_features,
            on=["game_date", "dh", "player_id"],
            how="inner",
            suffixes=("", "_bf"),
            validate="1:m"
        )

        lineups_with_stats = lineups_with_stats.drop(columns=["team_bf"], errors="ignore")

        agg_cols = {
            "ops_rolling": ["mean"],
            "wrc_plus_rolling": ["mean"],
            "woba_rolling": ["mean"],
            "babip_rolling": ["mean"],
            "bb_k_rolling": ["mean"],
            "barrel_percent_rolling": ["mean"],
            "hard_hit_rolling": ["mean"],
            "ev_rolling": ["mean"],
            "iso_rolling": ["mean"],
            "gb_fb_rolling": ["mean"],
            "baserunning_rolling": ["mean"],
            "wraa_rolling": ["mean"],
            "wpa_rolling": ["mean"]
        }

        team_features = (
            lineups_with_stats
            .groupby(["game_id", "game_date", "team", "opposing_team", "dh", "window_size"])
            .agg(agg_cols)
        )

        team_features.columns = [col[0] for col in team_features.columns]
        team_features = (
            team_features.reset_index().pivot(index=["game_id", 'game_date', 'team', 'opposing_team', 'dh'], columns='window_size')
        )

        team_features.columns = [
            f"{c0}_w{c1}" for (c0, c1) in team_features.columns.to_flat_index()
        ]

        return team_features

    
    def _get_batting_features(self, schedule_df: DataFrame, force_recreate: bool = False) -> DataFrame:
        """Get batting features for all games efficiently"""
        raw_batter_data = self._load_batting_data()
        lineups_data = self._load_lineups_data()
        lineups_data['game_date'] = pd.to_datetime(lineups_data['game_date'])
        batting_features = BattingFeatures(self.season, raw_batter_data)
        
        self.logger.info(f" Calculating batting rolling stats for {self.season}")
        batting_features = batting_features.calculate_all_player_rolling_stats(force_recreate)

        self.logger.info(f" Merging schedule, lineups, and batting rolling stats for {self.season}")
        team_features = self._merge_schedule_with_batting_features(schedule_df, lineups_data, batting_features)

        self.logger.info(f" Adding opposing team batting stats to each row for {self.season}")
        team_and_opponent_feats = self._add_opponent_features(team_features)
        team_and_opponent_feats = team_and_opponent_feats.sort_index(level=['game_date', 'dh', 'team'])
        team_and_opponent_feats = team_and_opponent_feats.reset_index()

        return team_and_opponent_feats

                                
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

        if df.index.has_duplicates:
            raise ValueError("Input index has duplicates; disambiguate (e.g., include dh/game_datetime).")

        opp_view = df[feature_cols].copy()
        opp_view.index = opp_view.index.swaplevel(team_level, opp_level)
        opp_view = opp_view.sort_index()

        opp_aligned = opp_view.reindex(df.index)
        opp_aligned.columns = [f"opposing_{c}" for c in feature_cols]

        return pd.concat([df, opp_aligned], axis=1)    

    def _load_pitching_features(self) -> DataFrame:
        pitching_features = PitchingFeatures(self.season).load_data()
        return pitching_features
    
    def _load_fielding_features(self) -> DataFrame:
        fielding_features = FieldingFeatures(self.season).load_data()
        return fielding_features
    

    def _match_schedule_to_odds(self, schedule_data: DataFrame, odds_data: DataFrame) -> DataFrame:
        """
        Match schedule games to odds. A single game will have many odds matches. Games without a match are due to incorrect labeling
        of game_datetime in the schedule data, so the odds game_datetime is used. This is handled in find_unmatched_games and _handle_unmatched_games
        """

        def find_unmatched_games(merged_games: DataFrame, odds_data: DataFrame, keys: List[str] = None) -> DataFrame:
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

        games_without_odds = merged_games[
                                merged_games['away_opening_odds'].isna() & 
                                merged_games['away_current_odds'].isna() & 
                                merged_games['home_opening_odds'].isna() &
                                merged_games['home_current_odds'].isna()
                            ]
        
        self.logger.info(f" After basic merge: {len(merged_games)} rows")
        self.logger.info(f" Unique games in schedule: {schedule_data[['game_date', 'dh', 'away_team', 'home_team']].drop_duplicates().shape[0]}")
        self.logger.info(f" Unique games in odds: {odds_data[['game_date', 'game_datetime', 'away_team', 'home_team']].drop_duplicates().shape[0]}")
        self.logger.info(f" Games without odds: {len(games_without_odds)}")

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
                               if col.endswith('_odds') and not col in ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']]
                
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
                    suffixes=('', '_odds')
                )

                final_merged = final_merged.drop_duplicates(
                    subset=['game_date', 'game_id', 'away_team', 'home_team', 'dh']
                )
                
                cols_to_drop = [col for col in final_merged.columns 
                               if col.endswith('_odds') and not col in ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']]
                
                final_merged = final_merged.drop(columns=cols_to_drop)
                final_merged = final_merged.set_index(['game_date', 'dh', 'game_datetime', 'away_team', 'home_team'])

                
                self.logger.info(f" Merged games: {len(final_merged)}")
                
                original_odds_count = len(odds_data)
                all_odds_count = len(all_odds_data)
                self.logger.info(f" Original odds: {original_odds_count}, All merged odds (including reconciled): {all_odds_count}")
                
                if all_odds_count > original_odds_count:
                    self.logger.warning(f" Merged games ({all_odds_count}) exceed total available odds ({original_odds_count})!")
                else:
                    self.logger.info(f" Merge validation passed: {all_odds_count} merged <= {original_odds_count} total odds")

                # self._print_matching_summary(schedule_data, odds_data, final_merged, remaining_unmatched_games)
                merged_games = final_merged
            else:
                self.logger.warning("handle_unmatched_games returned no matches")

        id_cols = ["game_date", "dh", "game_datetime", 'away_team', 'home_team']

        merged_games = merged_games.reset_index()
        merged_games = merged_games.sort_values(id_cols).set_index(id_cols)
        
        game_metadata_cols = [
            'game_id', 'day_night_game', 'season', 'venue_name', 'venue_id',
            'venue_elevation', 'venue_timezone', 'venue_gametime_offset', 'status',
            'away_probable_pitcher', 'home_probable_pitcher', 'away_starter_normalized',
            'home_starter_normalized', 'wind', 'condition', 'temp', 'away_score',
            'home_score', 'winning_team', 'losing_team', 'away_starter', 'home_starter',
            'winner'
        ]
        
        odds_cols = ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']
        
        existing_metadata_cols = [col for col in game_metadata_cols if col in merged_games.columns]
        existing_odds_cols = [col for col in odds_cols if col in merged_games.columns]
        metadata_df = merged_games.groupby(level=id_cols)[existing_metadata_cols].first()
        
        if existing_odds_cols:
            odds_df = merged_games[existing_odds_cols + ['sportsbook']].reset_index()
            odds_pivoted = odds_df.pivot_table(
                index=id_cols,
                columns='sportsbook',
                values=existing_odds_cols,
                aggfunc='first'
            )
            
            if odds_pivoted.columns.nlevels > 1:
                odds_pivoted.columns = [f"{col}_{sportsbook}" for col, sportsbook in odds_pivoted.columns]
            
            final_result = metadata_df.join(odds_pivoted, how='left')
        else:
            final_result = metadata_df
            self.logger.debug("="*50 + "\n")
            self.logger.debug(" Resulting DataFrame after matching schedule")       
            self.logger.debug(final_result.to_string())
            self.logger.debug("="*50 + "\n")

        return final_result

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

        
    
    def start_pipeline(self, force_recreate: bool = False, clear_log: bool = False):
        # Set up args if not already done (for script compatibility)
        global args
        if args is None:
            import argparse
            parser = argparse.ArgumentParser(description="Feature engineering runner")
            parser.add_argument("--force_recreate", action="store_true", help="Recreate batting rolling features, even if cached file exists")
            parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
            parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
            parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
            args = parser.parse_args([])  # Parse empty args for programmatic use
            args.force_recreate = force_recreate
            args.clear_log = clear_log

        self.logger.info("="*60)
        self.logger.info(f" Starting feature pipeline...")
        self.logger.info("="*60 + "\n")
        schedule_data = self._load_schedule_data()

        odds_data = self._load_odds_data()
        odds_sch_matched = self._match_schedule_to_odds(schedule_data, odds_data)

        transformed_schedule = self._transform_schedule(odds_sch_matched)
        batting_features = self._get_batting_features(transformed_schedule, args.force_recreate)
        context_features = GameContextFeatures(schedule_data, self.season).load_features()
        context_features = context_features.drop(columns=['away_team', 'home_team'])

        transformed_schedule_reset = transformed_schedule.reset_index()
        batting_features_reset = batting_features.reset_index()
        
        sch_batting_features = pd.merge(
            transformed_schedule_reset,
            batting_features_reset,
            on=['game_date', 'game_id', 'dh', 'team', 'opposing_team'],
            how='inner',
            validate='1:1' 
        )

        final_features = pd.merge(
            sch_batting_features,
            context_features,
            on=['game_id', 'dh', 'game_date', 'game_datetime'],
            how='inner',
            validate='m:1',
            suffixes=('_sch_bat', '')
        )

        final_features = final_features.drop(columns=[col for col in final_features.columns if col.endswith('_sch_bat') or col in ['wind', 'condition']])
        final_features = final_features.set_index(['game_date', 'dh', 'team'])
        
        self.logger.info(f" Final merged dataset shape: {final_features.shape}")

        self.logger.debug("="*60 + "\n")
        self.logger.debug(" Final features DataFrame tail")
        self.logger.debug(final_features.tail(10).to_string())
        self.logger.debug("="*60 + "\n")

        return final_features
        
def main():
    create_args()
    
    feat_pipe = FeaturePipeline(2021)
    features = feat_pipe.start_pipeline()


if __name__ == "__main__":
    main()

    
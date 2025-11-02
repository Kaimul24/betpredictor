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

from src.config import PROJECT_ROOT

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
from src.utils import setup_logging

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
    args = parser.parse_args()
    return args

class FeaturePipeline:

    def __init__(self, season: int, logger: Optional[logging.Logger] = None):
        self.season = season
        self.cache = {}
        self.logger = logger or logging.getLogger("feature_pipeline")

    def _transform_schedule(self, schedule_data: DataFrame) -> DataFrame:
        """Splits each game in the schedule into 2 rows, each representing one team's perspective of the game"""
        self.logger.info(f" Transforming schedule for {self.season}")

        schedule_data = schedule_data.reset_index().copy()

        unique_cols = ['away_team', 'home_team', 'away_probable_pitcher', 'home_probable_pitcher', 'starter_normalized', 'opposing_starter_normalized', 'away_score', 'home_score']
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
    
    def _merge_schedule_with_batting_features(self, schedule_df: DataFrame, lineups_data: DataFrame, batting_features: DataFrame) -> DataFrame:
        """
        Merges schedule, lineups, and rolling batting features into one DataFrame.
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

        batting_features['player_id'] = batting_features['player_id'].astype('int64')
        lineups_for_games['player_id'] = lineups_for_games['player_id'].astype('int64')

        # print(f"LINEUP FOR GAMES: {len(lineups_for_games)}")
        lineups_with_stats = lineups_for_games.merge(
            batting_features,
            on=["game_date", "dh", "player_id"],
            how="left",
            suffixes=("", "_bf"),
        )

        # print(f"AFTER MERGES: {len(lineups_with_stats)}")
        # print(lineups_with_stats.isna().any(axis=1).sum())

        lineups_with_stats = lineups_with_stats.drop(columns=["team_bf"], errors="ignore")

        value_cols = [c for c in lineups_with_stats.columns 
              if any(s in c for s in ['_season', '_ewm_h3', '_ewm_h10', '_ewm_h25']) or
              c == 'team_frv_per_9']


        league_medians = batting_features[value_cols].median()
        lineups_with_stats[value_cols] = lineups_with_stats[value_cols].fillna(league_medians)

        team_features = (
            lineups_with_stats
            .groupby(["game_id", "game_date", "team", "opposing_team", "dh"], observed=True)[value_cols]
            .mean()
        )

        return team_features
    
    def _get_batting_features(self, schedule_df: DataFrame, force_recreate: bool = False) -> DataFrame:
        """Get batting features for all games efficiently"""
        raw_batter_data = self._load_batting_data()
        raw_batter_data = raw_batter_data[~((raw_batter_data['pos'].str.startswith('P')) & (raw_batter_data['player_id'] != 19755))] # Shohei

        lineups_data = self._load_lineups_data()
        lineups_data['game_date'] = pd.to_datetime(lineups_data['game_date'])
        lineups_data = lineups_data[~lineups_data['position'].str.startswith('P')]

        batting_features = BattingFeatures(self.season, raw_batter_data, force_recreate)
        
        self.logger.info(f" Calculating batting rolling stats for {self.season}")
        batting_features = batting_features.load_features()

        raw_fielding_data = self._load_fielding_data()
        fielding_feats = FieldingFeatures(self.season, raw_fielding_data).load_features()

        self.logger.info(f" Merging batting rolling and fielding stats for {self.season}")
        bat_fld_feats = self._merge_batting_fielding_features(batting_features, fielding_feats)

        base_stat_cols = ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 'k_percent', 'bb_percent', 'barrel_percent', 'hard_hit', 
                          'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa', 'frv_per_9']
        
        rolling_stat_cols = [col for col in bat_fld_feats.columns if any(base_name in col for base_name in base_stat_cols)]
        rename_cols = {col: f"team_{col}" for col in rolling_stat_cols}
        bat_fld_feats = bat_fld_feats.rename(columns=rename_cols)

        bat_fld_feats['team_frv_per_9'] = bat_fld_feats['team_frv_per_9'].fillna(0.0)

        self.logger.info(f" Merging schedule, lineups, and batting/fielding rolling stats for {self.season}")
        team_features = self._merge_schedule_with_batting_features(schedule_df, lineups_data, bat_fld_feats)

        self.logger.info(f" Adding opposing team batting stats to each row for {self.season}")
        team_and_opponent_feats = self._add_opponent_features(team_features)
        team_and_opponent_feats = team_and_opponent_feats.sort_index(level=['game_date', 'dh', 'team'])
        team_and_opponent_feats = team_and_opponent_feats.reset_index()


        self.logger.info(f" Adding team batting comparison stats for {self.season}")
        batting_comp_cols = ["woba", "wrc_plus", "hard_hit", "barrel_percent", "bb_k", "ops",
                             "babip", "ev", "iso", "baserunning", "wpa"]
        ewm_cols = ['season', 'ewm_h3', 'ewm_h10', 'ewm_h25']
        batting_comp_stats = FeaturePipeline._add_matchup_cols_diff_same_base(team_and_opponent_feats, batting_comp_cols, ewm_cols)
        team_and_opponent_feats = team_and_opponent_feats.assign(**batting_comp_stats)

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
                all_odds_data.set_index(["game_date", "dh", "home_team", "away_team", "game_id"], inplace=True)
                all_odds_data.drop(columns=[col for col in all_odds_data.columns if col not in ['sportsbook', 'away_opening_prob_raw', 'home_opening_prob_raw', 'home_opening_prob_nv', 'away_opening_prob_nv']], inplace=True)
            else:
                self.logger.warning("handle_unmatched_games returned no matches")

        ## REFACTOR LATER
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
        
    def start_pipeline(self, force_recreate: bool = False, clear_log: bool = False) -> Tuple[DataFrame, DataFrame]:
        self.logger.info("="*60)
        self.logger.info(f" Starting feature pipeline...")
        self.logger.info("="*60 + "\n")
        schedule_data = self._load_schedule_data()

        far_future_games = schedule_data[(schedule_data['game_datetime'].astype('datetime64[ns]') - schedule_data['game_date'].astype('datetime64[ns]')) > pd.Timedelta("2 Days")]

        self.logger.info(f" Far rescheduled games\n{far_future_games}")
        self.logger.info(f" Dropping rescheduled games that are far past original game_date")

        schedule_data = schedule_data.drop(far_future_games.index)

        odds_data = self._load_odds_data()
        odds_feats = Odds(odds_data, self.season).load_features()
        odds_sch_matched, raw_odds_data = self._match_schedule_to_odds(schedule_data, odds_feats)

        transformed_schedule = self._transform_schedule(odds_sch_matched)
        batting_features = self._get_batting_features(transformed_schedule, force_recreate)
        
        context_features = GameContextFeatures(self.season, schedule_data).load_features()
        context_features = context_features.drop(columns=['away_team', 'home_team'])

        team_features = TeamFeatures(self.season, transformed_schedule).load_features()
        team_feat_cols = ['win_pct_season', 'win_pct_ewm_h3', 'win_pct_ewm_h8', 'win_pct_ewm_h20', 
                          'pyth_expectation_season', 'pyth_expectation_ewm_h3', 'pyth_expectation_ewm_h8', 'pyth_expectation_ewm_h20', 
                          'run_diff_season', 'run_diff_ewm_h3', 'run_diff_ewm_h8', 'run_diff_ewm_h20',
                          'one_run_win_pct_season', 'one_run_win_pct_ewm_h3', 'one_run_win_pct_ewm_h8', 'one_run_win_pct_ewm_h20']
        
        team_features.rename(columns = {col: f"team_{col}" for col in team_feat_cols}, inplace=True)
        team_features_with_opp = self._add_opponent_features(team_features, feature_cols=[f"team_{col}" for col in team_feat_cols])
        
        team_features_with_opp = team_features_with_opp.sort_index(level=['game_date', 'dh', 'team'])
        team_features_with_opp.rename(columns = {col: f"team_{col}" for col in team_feat_cols}, inplace=True)

        raw_pitching_data = self._load_pitching_data()
        pitching_features = PitchingFeatures(self.season, raw_pitching_data, force_recreate).load_features().reset_index()
        pitching_features = pitching_features.set_index(['game_date', 'dh', 'team', 'opposing_team'])

        self.logger.info(f" Adding opposing team bullpen pitching stats to each row for {self.season}")
        bullpen_cols = [col for col in pitching_features.columns if 'pen_' in col and not col.startswith('opposing_')]

        pitching_features_opp_bp = self._add_opponent_features(pitching_features, feature_cols=bullpen_cols)
        pitching_features_opp_bp = pitching_features_opp_bp.sort_index(level=['game_date', 'dh', 'team'])
        pitching_features_opp_bp = pitching_features_opp_bp.reset_index()

        transformed_schedule_reset = transformed_schedule.reset_index()
        batting_features_reset = batting_features.reset_index()
        
        sch_batting_features = pd.merge(
            transformed_schedule_reset,
            batting_features_reset,
            on=['game_date', 'game_id', 'dh', 'team', 'opposing_team'],
            how='inner',
            validate='1:1' 
        )

        sch_bat_ctx = pd.merge(
            sch_batting_features,
            context_features,
            on=['game_id', 'game_date', 'dh', 'game_datetime'],
            how='inner',
            validate='m:1',
            suffixes=('_sch_bat', '')
        )
        
        sch_bat_ctx_team = pd.merge(
            sch_bat_ctx,
            team_features_with_opp,
            on=['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team'],
            how='inner',
            validate='1:1',
        )

        final_features = pd.merge(
            sch_bat_ctx_team,
            pitching_features_opp_bp,
            on=['game_date', 'dh', 'team', 'opposing_team', 'season'],
            how='inner',
            validate='1:1',
            suffixes=('_to_drop', '_to_drop')
        )

        self.logger.info(f" Adding engineered matchup columns...")

        pitcher_ewm_cols = ['season', 'ewm_h3', 'ewm_h8', 'ewm_h20']
        starter_cols = ['starter_era', 'starter_babip', 'starter_hard_hit', 'starter_k_percent', 
                        'starter_barrel_percent', 'starter_fip', 'starter_siera', 'starter_stuff',
                        'starter_ev', 'starter_hr_fb', 'starter_wpa']
        
        starter_matchups = FeaturePipeline._add_matchup_cols_diff_same_base(df=final_features,
                                                                            cols=starter_cols,
                                                                            ewm_cols=pitcher_ewm_cols)
                                                                            
        final_features = final_features.assign(**starter_matchups)

        pen_cols = ['pen_era', 'pen_babip', 'pen_hard_hit', 'pen_k_percent', 
                        'pen_barrel_percent', 'pen_fip', 'pen_siera', 'pen_stuff',
                        'pen_ev', 'pen_hr_fb', 'pen_wpa_li']
        
        
        pen_matchups = FeaturePipeline._add_matchup_cols_diff_same_base(df=final_features,
                                                                            cols=pen_cols,
                                                                            ewm_cols=pitcher_ewm_cols)
                                                                            
        final_features = final_features.assign(**pen_matchups)

        team_pitch_cols = [ "starter_fip", "starter_k_percent", "starter_bb_percent", "starter_barrel_percent"]

        opp_team_bat_cols = ["woba", "k_percent", "bb_percent", "barrel_percent"]
        bat_ewm_cols = ['season', 'ewm_h3', 'ewm_h10', 'ewm_h25']
        team_pitching_vs_opp_batting = FeaturePipeline._add_matchup_cols_diff_base(
                                                                        df=final_features,
                                                                        col1=team_pitch_cols,
                                                                        col2=opp_team_bat_cols,
                                                                        col1_ewm_cols=pitcher_ewm_cols,
                                                                        col2_ewm_cols=bat_ewm_cols)   
        
        final_features = final_features.assign(**team_pitching_vs_opp_batting)

        team_metrics_cols = ["win_pct", "pyth_expectation", "run_diff", "one_run_win_pct"]
        team_metrics_ewm_cols = ['season', 'ewm_h3', 'ewm_h8', 'ewm_h20']
        team_metrics_matchups = FeaturePipeline._add_matchup_cols_diff_same_base(df=final_features,
                                                                                 cols=team_metrics_cols,
                                                                                 ewm_cols=team_metrics_ewm_cols)
        
        final_features = final_features.assign(**team_metrics_matchups)

        assert (final_features['starter_fip_woba_season_diff'] == final_features['team_starter_fip_season'] - final_features['opposing_team_woba_season']).all()


        final_features = final_features.drop(columns=[col for col in final_features.columns if col.endswith('_sch_bat') or
                                                    col in ['wind', 'condition'] or 
                                                    col.endswith('_to_drop')])
        
        final_features = final_features.set_index(['season', 'game_date', 'dh', 'team', 'opposing_team', 'game_id'])

        assert final_features.index.get_level_values('game_date').is_monotonic_increasing, "final_features game_date not globally sorted"

        final_features.drop(columns=['team_starter_last_app_date', 'opposing_team_starter_last_app_date'], inplace=True) ## FIX
        final_features = self._collapse_data(final_features)

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
    
    def _collapse_data(self, df: DataFrame) -> DataFrame:
        df = df.copy()

        home_df = df[df['is_home']]
        away_df = df[~df['is_home']]
        
        const_cols = [col for col in df.columns if 'team_' not in col and 'opposing_' not in col]
        
        assert len(home_df) + len(away_df) == len(df)

        home_rename = {
            col: f"home_{col}" for col in home_df.columns if col not in const_cols and 'opposing_team_' not in col and 'diff' not in col
        }

        away_rename = {
            col: f"away_{col}" for col in away_df.columns if col not in const_cols and 'opposing_team_' not in col and 'diff' not in col
        }

        home_df = home_df.rename(columns=home_rename)
        away_df = away_df.rename(columns=away_rename)

        home_df.index = home_df.index.set_names('home_team', level='team')
        away_df.index = away_df.index.set_names('away_team', level='team')

        home_df = home_df.reset_index(level='opposing_team').drop(columns='opposing_team')
        away_df = away_df.reset_index(level='opposing_team').drop(columns='opposing_team')

        home_df = home_df.drop(columns=[col for col in home_df.columns if col.startswith("opposing_team_")])
        away_df = away_df.drop(columns=[col for col in away_df.columns if col.startswith("opposing_team_")])
        
        assert 'opposing_team' not in away_df.columns and 'opposing_team' not in away_df.index.names

        final_df = pd.merge(
            home_df,
            away_df,
            how='inner',
            left_index=True,
            right_index=True,
            suffixes=('', '_drop_away')

        )

        from_away_cols = [col for col in final_df.columns if col.endswith('_drop_away')]
        final_df = final_df.drop(columns=from_away_cols)
        final_df = final_df.rename(columns={'is_winner': 'is_winner_home'})
        
        return final_df


    @staticmethod
    def _add_matchup_cols_diff_same_base(df: DataFrame, cols: List[str], ewm_cols: List[str]) -> Dict[str, pd.Series]:
        
        result = {}

        for col in cols:
            for c in ewm_cols:
                result[f"{col}_{c}_diff"] = df[f"team_{col}_{c}"] - df[f"opposing_team_{col}_{c}"]

        return result
    
    @staticmethod
    def _add_matchup_cols_diff_base(df: DataFrame, col1: List[str], col2: List[str], col1_ewm_cols: List[str], col2_ewm_cols: List[str]) -> Dict[str, pd.Series]:
        if len(col1) != len(col2) or len(col1_ewm_cols) != len(col2_ewm_cols):
            raise ValueError("Col1 and Col2 must be same length.")
        
        result = {}
        for c1, c2 in zip(col1, col2):
            for e1, e2 in zip(col1_ewm_cols, col2_ewm_cols):
                result[f"{c1}_{c2}_{e1}_diff"] = df[f"team_{c1}_{e1}"] - df[f"opposing_team_{c2}_{e2}"]

        return result
    
        
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
        lineups_data = loader.load_lineup_players(self.season)
        return lineups_data
    
    def _load_pitching_data(self) -> DataFrame:
        loader = PlayerLoader()
        pitching_data = loader.load_for_season_pitcher(self.season)
        return pitching_data
    
    def _load_fielding_data(self) -> DataFrame:
        loader = PlayerLoader()
        fielding_data = loader.load_fielding_stats(self.season)
        return fielding_data
        
def main():
    args = create_args()
    logger = setup_logging("feature_pipeline", LOG_FILE, args=args)
    
    feat_pipe = FeaturePipeline(2021, logger)

    features, raw_odds_data = feat_pipe.start_pipeline(args.force_recreate, args.clear_log)

    # with open("raw_odds_data.txt", "w") as f:
    #     f.write(raw_odds_data.to_string())

if __name__ == "__main__":
    main()
    

"""
Orchestrates the feature engineering process for all features. 
"""

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from fuzzywuzzy import fuzz
import logging

from data.features.game_features.context import GameContextFeatures
from data.features.game_features.matchup import MatchupFeatures
from data.features.game_features.odds import Odds
from data.features.player_features.batting import BattingFeatures
from data.features.player_features.fielding import FieldingFeatures
from data.features.player_features.pitching import PitchingFeatures
from data.features.player_features.war import WAR
from data.features.team_features.schedule import ScheduleFeatures


class FeaturePipeline():

    def __init__(self, season: int):
        self.season = season

    def _load_schedule(self) -> DataFrame:
        schedule_data = ScheduleFeatures(self.season).load_data()
        return schedule_data
    
    def _load_odds(self) -> DataFrame:
        odds_data = Odds(self.season).load_data()
        return odds_data
    
    def _load_batting_features(self) -> DataFrame:
        batting_features = BattingFeatures(self.season).load_data()
        return batting_features
    
    def _load_pitching_features(self) -> DataFrame:
        pitching_features = PitchingFeatures(self.season).load_data()
        return pitching_features
    
    def _load_fielding_features(self) -> DataFrame:
        fielding_features = FieldingFeatures(self.season).load_data()
        return fielding_features
    


    
    def _calculate_fuzzy_match_score(self, name1: str, name2: str, threshold: int = 80) -> float:
        """
        Calculate fuzzy matching score between two names.
        Returns 1.0 for exact match, 0.5 for fuzzy match above threshold, 0 otherwise.
        """
        if pd.isna(name1) or pd.isna(name2):
            return 0.0
        if name1 == name2:
            return 1.0
        
        ratio = fuzz.ratio(name1, name2)
        if ratio >= threshold:
            return 0.5  # Fuzzy match gets half score
        return 0.0
    
    def _match_schedule_to_odds(self, schedule_data: DataFrame, odds_data: DataFrame) -> DataFrame:
        """
        Match schedule games to odds with robust handling of doubleheaders, starter name errors, and fuzzy matching.
        """

        keys = ['game_date', 'game_datetime', 'away_team', 'home_team']

        merged_games = pd.merge(
            schedule_data,
            odds_data,
            how='inner',
            on=keys,
            suffixes=('_sch', '_odds')
        )

        print(f"After basic merge: {len(merged_games)} rows")

        unmatched_games = odds_data.merge(
            merged_games[keys], how='left', on=keys, indicator=True
        )

        unmatched_games = unmatched_games[unmatched_games['_merge'] == 'left_only']

        print(unmatched_games[['game_date', 'game_datetime', 'away_team', 'home_team']])

        print(f"Unmatched Games: {len(unmatched_games)}")
        
        print(unmatched_games.head(3))
        return merged_games
    
    
    def start_pipeline(self):
        schedule_data = self._load_schedule()
        print("SCH: ", len(schedule_data))
        odds_data = self._load_odds()
        print("ODDS: ", len(odds_data))

        games_df = self._match_schedule_to_odds(schedule_data, odds_data)
        print(len(games_df))
        return games_df
        
def main():
    feat_pipe = FeaturePipeline(2021)
    games = feat_pipe.start_pipeline()
    odds = feat_pipe._load_odds()
    



if __name__ == "__main__":
    main()

    
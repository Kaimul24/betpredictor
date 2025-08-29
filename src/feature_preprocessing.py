from src.data.features.feature_pipeline import FeaturePipeline

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import logging, sys, argparse
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class PreProcessing():
    def __init__(self, seasons: List[int]):
        self.seasons = seasons

        self.target = ['is_winner',]

        self.exclude_columns = [
            # Game outcome information (data leakage)
            'team_score', 'opposing_team_score', 'home_score', 
            'winning_team', 'losing_team',
            
            # Identifiers and metadata
            'game_id', 'game_datetime', 'season',
            
            # Player names (categorical with too many unique values)
            'away_starter_normalized', 'home_starter_normalized',
            'starter_normalized', 'opposing_starter_normalized',
            'name_team_starter', 'normalized_player_name_team_starter',
            'name_opposing_starter', 'normalized_player_name_opposing_starter',
            
            # IDs and foreign keys
            'team_starter_id', 'opposing_starter_id', 'player_id_team_starter',
            'player_id_opposing_starter', 'mlb_id', 'venue_id',
            
            # Date columns (temporal leakage risk)
            'last_app_date_team_starter', 'last_app_date_opposing_starter',
            
            # Status and other metadata
            'status', 'venue_name', 'venue_timezone', 'venue_gametime_offset'
        ]

    def _feature_scaling(self, dfs: List[Tuple[DataFrame, DataFrame]]):
        seasons_length = len(self.seasons)
        
        stat_dfs = [df[0] for df in dfs]
        odds_dfs = [df[1] for df in dfs]
        
        filtered_dfs = [df[[col for col in df.columns if col not in self.exclude_columns]] for df in stat_dfs]
        
        train_dfs = filtered_dfs[:seasons_length-1]
        test_val_dfs = filtered_dfs[seasons_length-1:]

        print(train_dfs[0].dtypes.value_counts())

        all_na_rows = []
        for i, df in enumerate(train_dfs):
            na_rows = df[df.isna().any(axis=1)]
            all_na_rows.append(f"--- DataFrame {i} ---\n{na_rows.to_string()}")

        with open('nan_rows.txt', 'w') as f:
            f.write('\n\n'.join(all_na_rows))

        return filtered_dfs, odds_dfs

    def _separate_odds_cols(self, dfs: List[DataFrame]) -> List[Tuple[DataFrame, DataFrame]]:
        results = []

        for df in dfs:
            odds_cols = [col for col in df.columns if 'odds' in col]
            
            feat_df = df[[col for col in df.columns if col not in odds_cols]]
            odds_df = df[odds_cols] if odds_cols else DataFrame(index=df.index)
            
            results.append((feat_df, odds_df))
        
        return results
        
    def _get_features(self) -> List[DataFrame]:
        all_features = []

        for year in self.seasons:
            feat_pipe = FeaturePipeline(year)
            season_feats = feat_pipe.start_pipeline(force_recreate=True)
            all_features.append(season_feats)
        
        return all_features
    
    def preprocess_feats(self):
        raw_feats = self._get_features()
        odds_separated_dfs = self._separate_odds_cols(raw_feats)

        stats_dfs, odds_dfs = self._feature_scaling(odds_separated_dfs)
        
        return stats_dfs


def main():
    pre_processor = PreProcessing([2021, 2022, 2023, 2024])
    preprocessed_feats = pre_processor.preprocess_feats()
    #print(preprocessed_feats)

if __name__ == "__main__":
    main()
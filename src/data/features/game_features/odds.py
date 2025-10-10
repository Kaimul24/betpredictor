"""
Handles construction of odds features and converting to implied probabilities.
"""

from src.data.loaders.odds_loader import OddsLoader
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Tuple
import logging
import pandas as pd
from scipy.special import logit
import math

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Odds(BaseFeatures):

    def __init__(self, data: DataFrame, season: int) -> None:
        super().__init__(season, data)

        self.base_odds_cols = ['away_opening_odds', 'home_opening_odds']

    def load_features(self):
        df = self.data[['game_date', 'game_datetime', 'away_team', 'home_team', 'winner', 'sportsbook', 'away_opening_odds', 
                        'home_opening_odds']].copy()

        df = self._convert_imp_prob(df)
        df = self._remove_vig(df)
        df = self._build_odds_feats_per_game(df)
        
        return df

    def _build_odds_feats_per_game(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        g = df.groupby(['game_date', 'game_datetime', 'away_team', 'home_team'])

        prob_medians_raw = g[['home_opening_prob_raw', 'away_opening_prob_raw']].transform('median')
        prob_means_raw = g[['home_opening_prob_raw', 'away_opening_prob_raw']].transform('mean')
        prob_std_raw = g[['home_opening_prob_raw', 'away_opening_prob_raw']].transform('std')

        df['p_open_home_median'] = prob_medians_raw['home_opening_prob_raw']
        df['p_open_home_mean'] = prob_means_raw['home_opening_prob_raw']
        df['p_open_home_std'] = prob_std_raw['home_opening_prob_raw'].fillna(0.0)

        df['p_open_away_median'] = prob_medians_raw['away_opening_prob_raw']
        df['p_open_away_mean'] = prob_means_raw['away_opening_prob_raw']
        df['p_open_away_std'] = prob_std_raw['away_opening_prob_raw'].fillna(0.0)
        

        prob_medians_nv = g[['home_opening_prob_nv', 'away_opening_prob_nv']].transform('median')
        prob_means_nv = g[['home_opening_prob_nv', 'away_opening_prob_nv']].transform('mean')
        prob_std_nv = g[['home_opening_prob_nv', 'away_opening_prob_nv']].transform('std')

        df['p_open_home_max_nv'] = g['home_opening_prob_nv'].transform('max')
        df['p_open_home_min_nv'] = g['home_opening_prob_nv'].transform('min')

        df['p_open_away_max_nv'] = g['away_opening_prob_nv'].transform('max')
        df['p_open_away_min_nv'] = g['away_opening_prob_nv'].transform('min')

        df['p_open_home_median_nv'] = prob_medians_nv['home_opening_prob_nv']
        df['p_open_home_mean_nv'] = prob_means_nv['home_opening_prob_nv']
        df['p_open_home_std_nv'] = prob_std_nv['home_opening_prob_nv'].fillna(0.0)

        df['p_open_away_median_nv'] = prob_medians_nv['away_opening_prob_nv']
        df['p_open_away_mean_nv'] = prob_means_nv['away_opening_prob_nv']
        df['p_open_away_std_nv'] = prob_std_nv['away_opening_prob_nv'].fillna(0.0)

        df['p_open_mean_nv_diff'] = prob_medians_nv['home_opening_prob_nv'] - prob_medians_nv['away_opening_prob_nv']
        df['p_open_med_nv_diff'] = prob_means_nv['home_opening_prob_nv'] - prob_means_nv['away_opening_prob_nv']

        df['p_open_max_nv_diff'] = df['p_open_home_max_nv'] - df['p_open_away_max_nv']
        df['p_open_min_nv_diff'] = df['p_open_home_min_nv'] - df['p_open_away_min_nv']

        return df

    def _concat_all_odds_per_game(self, df: DataFrame) -> DataFrame:
        id_cols = ['game_date', 'game_datetime', 'away_team', 'home_team']
        
        all_odds_df = df.pivot(
            index=id_cols,
            columns='sportsbook',
            values = ['away_opening_odds', 'home_opening_odds']
        )

        all_odds_df.columns = [f"{col}_{sportsbook}" for col, sportsbook in all_odds_df.columns]

        return all_odds_df

    def _convert_imp_prob(self, df: DataFrame) -> DataFrame:
        
        def to_prob(line):
            line = line.astype(float)
            prob = np.where(line < 0, (-line) / (-line + 100.0), (100.0) / (line + 100.0))
            return prob
        
        for col in self.base_odds_cols:
            df[f"{col[:12]}_prob_raw"] = to_prob(df[col])

        return df
    
    def _remove_vig(self, data: DataFrame) -> DataFrame:

        def _no_vig(p_home: float, p_away: float) -> Tuple[float, float]:
            s = p_home + p_away
            return p_home / s, p_away / s
        
        
        data[f"home_opening_prob_nv"], data[f"away_opening_prob_nv"] = _no_vig(data["home_opening_prob_raw"], data["away_opening_prob_raw"])

        data["vig_open"] = data['home_opening_prob_raw'] + data['away_opening_prob_raw']
        return data
    
    
    
    def _handle_outliers(self, data: DataFrame) -> DataFrame:

        SENTINELS = {
            -100000,-90000,-85000,-75000,-50000,-35000,-30000,
            -25000,-20000,-10000,-9000,-8000,-7000,-6000,
            -5000,-4500,-4000,-3500,-3000,
            10000,9000,8000,7000,6000,5000,4500,4000,3500,3000,
            2500,2200,2000,1800,
        }



        data["is_sentinel"] = data["home_current_odds"].isin(SENTINELS) | \
                    data["away_current_odds"].isin(SENTINELS) 
        


        logger.info(f" Sentinel Rows: {data['is_sentinel'].sum()}")

        invalid_zero_odds_rows = ((data['away_opening_odds'] == 0.0) |
                     (data['home_opening_odds'] == 0.0) |
                     (data['away_current_odds'] == 0.0) |
                     (data['home_current_odds'] == 0.0))
        
        logger.info(f" Invalid rows (odds == 0.0)\n{data[invalid_zero_odds_rows][['game_date', 'game_datetime', 'away_team', 'home_team', 'away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']]}")
        logger.info(f" Removing {len(data[invalid_zero_odds_rows])} invalid rows...")

        valid_data = data[~invalid_zero_odds_rows].copy()

        valid_data['extreme_odds'] = ((valid_data['away_opening_prob_raw'] >= 0.99 )| 
                            (valid_data['away_current_prob_raw'] >= 0.99 )| 
                            (valid_data['home_opening_prob_raw'] >= 0.99 )|
                            (valid_data['home_current_prob_raw'] >= 0.99 )|
                            (valid_data['away_opening_prob_raw'] <= 0.01 )| 
                            (valid_data['away_current_prob_raw'] <= 0.01 )| 
                            (valid_data['home_opening_prob_raw'] <= 0.01 )|
                            (valid_data['home_current_prob_raw'] <= 0.01 ))

        with open('extreme_odds.txt', 'w') as f:
            f.write(valid_data[valid_data['extreme_odds']].to_string())

        extreme_odds = valid_data['extreme_odds'].sum()
        logger.info(f" Extreme odds: {extreme_odds}")
        logger.info(f" Maximum home_opening_prob_raw: {valid_data['home_opening_prob_raw'].max()}")

        valid_data = self._filter_cross_book_agreement(valid_data)

        valid_data['logit_movement_away'] = logit(valid_data['away_current_prob_nv']) - logit(valid_data['away_opening_prob_nv'])
        valid_data['logit_movement_home'] = logit(valid_data['home_current_prob_nv']) - logit(valid_data['home_opening_prob_nv'])

        valid_data['raw_sum_open'] = valid_data['home_opening_prob_raw'] + valid_data['away_opening_prob_raw']
        valid_data['raw_sum_curr'] = valid_data['home_current_prob_raw'] + valid_data['away_current_prob_raw']
        valid_data['raw_sum_oor'] = ((valid_data['raw_sum_open'] <= 1.005) |
                                               (valid_data['raw_sum_open'] >= 1.15) |
                                               (valid_data['raw_sum_curr'] <= 1.005) |
                                               (valid_data['raw_sum_curr'] >= 1.15) )
        
        logger.info(f" Raw sum out of range ([1.005, 1.15]]) rows: {valid_data['raw_sum_oor'].sum()}")

        with open('raw_sum_oor.txt', 'w') as f:
            f.write(valid_data[valid_data['raw_sum_oor']].to_string())
        

        df_sorted = valid_data.sort_values('home_opening_prob_raw', ascending=False)

        with open('away_current_prob_raw_sorted.txt', 'w') as f:
            f.write(df_sorted.to_string())

        return valid_data
        
    def _filter_cross_book_agreement(self, data: DataFrame) -> DataFrame:
        """
        Filter out current lines that don't agree with other sportsbooks for the same game.
        This helps identify live lines masquerading as closing lines.
        """
        # Group by game to analyze cross-book agreement
        game_groups = ['game_date', 'game_datetime', 'away_team', 'home_team']
        
        # Add flags for cross-book agreement
        data['is_valid_closing_line'] = True
        data['median_consensus_home'] = np.nan
        data['logit_deviation'] = np.nan
        data['num_other_books'] = 0
        
        # Group by game
        for game_key, game_df in data.groupby(game_groups):
            if len(game_df) < 2:
                # Single book - can't verify, mark as invalid by default
                data.loc[game_df.index, 'is_valid_closing_line'] = False
                data.loc[game_df.index, 'num_other_books'] = 0
                continue
            
            # For each sportsbook in this game
            for idx in game_df.index:
                current_book = data.loc[idx, 'sportsbook']
                current_home_prob = data.loc[idx, 'home_current_prob_nv']
                
                # Get other books for this game (excluding current book)
                other_books = game_df[game_df['sportsbook'] != current_book]
                
                if len(other_books) == 0:
                    data.loc[idx, 'is_valid_closing_line'] = False
                    data.loc[idx, 'num_other_books'] = 0
                    continue
                
                # Calculate median consensus from other books
                median_consensus = other_books['home_current_prob_nv'].median()
                data.loc[idx, 'median_consensus_home'] = median_consensus
                data.loc[idx, 'num_other_books'] = len(other_books)
                
                # Calculate logit deviation
                current_logit = logit(current_home_prob.clip(1e-6, 1-1e-6))
                consensus_logit = logit(median_consensus.clip(1e-6, 1-1e-6))
                logit_deviation = abs(current_logit - consensus_logit)
                data.loc[idx, 'logit_deviation'] = logit_deviation
                
            
                epsilon = 0.35 / (math.sqrt(max(2, len(other_books))))
                
                # Mark as invalid if deviation is too large
                if logit_deviation > epsilon:
                    data.loc[idx, 'is_valid_closing_line'] = False
        
        # Log results
        total_rows = len(data)
        invalid_closing_lines = (~data['is_valid_closing_line']).sum()
        logger.info(f" Cross-book agreement filter:")
        logger.info(f"   Total rows: {total_rows}")
        logger.info(f"   Invalid closing lines: {invalid_closing_lines} ({invalid_closing_lines/total_rows*100:.1f}%)")
        
        # Break down by reason
        single_book = (data['num_other_books'] == 0).sum()
        few_books = ((data['num_other_books'] == 1) & (~data['is_valid_closing_line'])).sum()
        high_deviation = ((data['num_other_books'] >= 2) & (~data['is_valid_closing_line'])).sum()
        
        logger.info(f"   Single book (no verification): {single_book}")
        logger.info(f"   Too few books (strict threshold): {few_books}")
        logger.info(f"   High deviation from consensus: {high_deviation}")
        
        # Save detailed analysis to file
        with open('cross_book_analysis.txt', 'w') as f:
            f.write("Cross-book Agreement Analysis\n")
            f.write("="*50 + "\n\n")
            
            # Invalid lines summary
            invalid_lines = data[~data['is_valid_closing_line']].copy()
            f.write(f"Invalid Closing Lines ({len(invalid_lines)} rows):\n")
            f.write(invalid_lines[['game_date', 'away_team', 'home_team', 'sportsbook', 
                                 'home_current_prob_nv', 'median_consensus_home', 
                                 'logit_deviation', 'num_other_books']].to_string())
            f.write("\n\n")
            
            # Games with high disagreement
            high_disagreement = data[data['logit_deviation'] > 0.5].copy() if 'logit_deviation' in data.columns else pd.DataFrame()
            if not high_disagreement.empty:
                f.write(f"Games with High Disagreement (>0.5 logit deviation):\n")
                f.write(high_disagreement[['game_date', 'away_team', 'home_team', 'sportsbook',
                                         'home_current_prob_nv', 'median_consensus_home', 
                                         'logit_deviation']].to_string())
        
        return data


        
def main():
    odds_loader = OddsLoader()
    odds_data = odds_loader.load_for_season(2021)
    odds = Odds(odds_data, 2021).load_features()
    with open('opening_odds_feats.txt', 'w') as f:
        f.write(odds.to_string())

if __name__ == "__main__":
    main()  
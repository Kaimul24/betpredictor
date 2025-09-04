"""
Handles construction of odds features and converting to implied probabilities.
"""

from src.data.loaders.odds_loader import OddsLoader
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Tuple
import logging
import pandas as pd

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Odds(BaseFeatures):

    def __init__(self, data: DataFrame, season: int) -> None:
        super().__init__(season, data)
        print(self.data.columns)
        print(self.data.head(6))

        self.base_odds_cols = ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']

    def load_features(self):
        df = self.data[['game_date', 'game_datetime', 'away_team', 'home_team', 'winner', 'sportsbook', 'away_opening_odds', 
                        'home_opening_odds', 'away_current_odds', 'home_current_odds']].copy()

        df = self._convert_imp_prob(df)
        df = self._remove_vig(df)
        df = self._handle_outliers(df)

        with open('all_odds.txt', 'w') as f:
            f.write(df.to_string())

        # df['away_opening_odds'] = df['away_opening_odds'].astype(int)
        # df['home_opening_odds'] = df['home_opening_odds'].astype(int)
        # df['away_current_odds'] = df['away_current_odds'].astype(int)
        # df['home_current_odds'] = df['home_current_odds'].astype(int)
        
        # df = df[(df['away_opening_odds'] == -10000) |
        #         (df['home_opening_odds'] == -10000) |
        #         (df['away_current_odds'] == -10000) |
        #         (df['home_current_odds'] == -10000)]
        
        #df = self._concat_all_odds_per_game(df)
        
    def _concat_all_odds_per_game(self, df: DataFrame) -> DataFrame:
        id_cols = ['game_date', 'game_datetime', 'away_team', 'home_team']
        
        all_odds_df = df.pivot(
            index=id_cols,
            columns='sportsbook',
            values = ['away_opening_odds', 'home_opening_odds', 'away_current_odds', 'home_current_odds']
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
        print(df.columns)
        return df
    
    def _remove_vig(self, data: DataFrame) -> DataFrame:

        def _no_vig(p_home: float, p_away: float) -> Tuple[float, float]:
            s = p_home + p_away
            return p_home / s, p_away / s
        
        
        data[f"home_opening_prob_nv"], data[f"away_opening_prob_nv"] = _no_vig(data["home_opening_prob_raw"], data["away_opening_prob_raw"])
        data[f"home_current_prob_nv"], data[f"away_current_prob_nv"] = _no_vig(data["home_current_prob_raw"], data["away_current_prob_raw"])

        data["vig_open"] = data['home_opening_prob_raw'] + data['away_opening_prob_raw']
        data["vig_current"] = data['home_current_prob_raw'] + data['away_current_prob_raw']

        return data
    
    def _handle_outliers(self, data: DataFrame) -> DataFrame:
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

       

        extreme_odds = valid_data['extreme_odds'].sum()
        logger.info(f" Extreme odds: {extreme_odds}")
        logger.info(f" Maximum home_opening_prob_raw: {valid_data['home_opening_prob_raw'].max()}")

        valid_data['raw_sum_open'] = valid_data['home_opening_prob_raw'] + valid_data['away_opening_prob_raw']
        valid_data['raw_sum_curr'] = valid_data['home_current_prob_raw'] + valid_data['away_current_prob_raw']
        valid_data['raw_sum_oor'] = ((valid_data['raw_sum_open'] <= 1.01) |
                                               (valid_data['raw_sum_open'] >= 1.15) |
                                               (valid_data['raw_sum_curr'] <= 1.01) |
                                               (valid_data['raw_sum_curr'] >= 1.15) )
        
        logger.info(f" Raw sum out of range ([1.01, 1.15]]) rows: {valid_data['raw_sum_oor'].sum()}")

        with open('raw_sum_oor', 'w') as f:
            f.write(valid_data[valid_data['raw_sum_oor']].to_string())
        

        df_sorted = valid_data.sort_values('home_opening_prob_raw', ascending=False)

        with open('away_current_prob_raw_sorted.txt', 'w') as f:
            f.write(df_sorted.to_string())
        return valid_data
        
        


        
def main():
    odds_loader = OddsLoader()
    odds_data = odds_loader.load_for_season(2021)
    odds = Odds(odds_data, 2021).load_features()

if __name__ == "__main__":
    main()  
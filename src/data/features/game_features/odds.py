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
        df['num_books'] = g['sportsbook'].transform('size')
        
        df['home_opening_logit_temp'] = logit(df['home_opening_prob_nv'].clip(1e-16, 1-1e-6))
        df['logit_prob_home_std_nv'] = (df.groupby(['game_date', 'game_datetime', 'away_team', 'home_team'])
                                        ['home_opening_logit_temp'].transform('std').fillna(0.0)
                                        )
        
        df = df.drop(columns=['home_opening_logit_temp'])
        prob_medians_nv = g[['home_opening_prob_nv', 'away_opening_prob_nv']].transform('median')

        df['p_open_home_median_nv'] = prob_medians_nv['home_opening_prob_nv']
    
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

        def _no_vig(p_home: pd.Series, p_away: pd.Series) -> Tuple[pd.Series, pd.Series]:
            s = p_home + p_away
            return p_home / s, p_away / s
        
        
        data[f"home_opening_prob_nv"], data[f"away_opening_prob_nv"] = _no_vig(data["home_opening_prob_raw"], data["away_opening_prob_raw"])

        data["vig_open"] = data['home_opening_prob_raw'] + data['away_opening_prob_raw']
        return data
        
def main():
    odds_loader = OddsLoader()
    odds_data = odds_loader.load_for_season(2021)
    odds = Odds(odds_data, 2021).load_features()
    with open('opening_odds_feats.txt', 'w') as f:
        f.write(odds.to_string())

if __name__ == "__main__":
    main()  
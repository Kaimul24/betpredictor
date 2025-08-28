"""
Handles fielding feature construction for use alone and in custom WAR calculation. Game level granularity cannot be obtained, 
so fielding value will be averaged over games in a month.
"""
import pandas as pd
import logging
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldingFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame):
        super().__init__(season, data)

    def load_features(self) -> DataFrame:
        frv_g_df = self._calc_frv_per_game_month()
        return frv_g_df[['player_id', 'month', 'frv_per_9']]

    def _calc_frv_per_game_month(self) -> DataFrame:
        """"Calcualtes the FRV per game per player for each month to be used as features."""
        df = self.data.copy()
        df = df.sort_values(['player_id', 'month'])
        df['frv_per_9'] = (df['frv'].div(df['total_innings'], fill_value=pd.NA)) * 9.0
        #logging.info(f"\n{df[['name', 'month', 'frv_g', 'frv', 'total_innings']].head(10)}")
        return df

if __name__ == "__main__":
    loader = PlayerLoader()
    fielding_stats = loader.load_fielding_stats(2021)
    f_feats = FieldingFeatures(2021, fielding_stats)
    f_feats = f_feats.load_features()
    print(f_feats)
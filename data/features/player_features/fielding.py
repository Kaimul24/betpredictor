"""
Handles fielding feature construction for use alone and in custom WAR calculation. Game level granularity cannot be obtained, 
so fielding value will be averaged over games in a month.
"""
import pandas as pd
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader

class FieldingFeatures(BaseFeatures):

    def __init__(self, season: int):
        super().__init__(season)

    def load_data(self) -> DataFrame:
        loader = PlayerLoader()
        df = loader.load_fielding_stats(self.season)

        if df.empty:
            print("DF is empty")
        
        return df
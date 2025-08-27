"""
Handles fielding feature construction for use alone and in custom WAR calculation. Game level granularity cannot be obtained, 
so fielding value will be averaged over games in a month.
"""
import pandas as pd
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader

class FieldingFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame):
        super().__init__(season, data)

    def load_features(self) -> DataFrame:
        pass

if __name__ == "__main__":
    loader = PlayerLoader()
    fielding_stats = loader.load_fielding_stats(2021)
    print(fielding_stats)
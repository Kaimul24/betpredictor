"""
Handles computation of batting stats features.
"""
import pandas as pd
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader
from src.config import DATES

from data.database import get_database_manager

class BattingFeatures(BaseFeatures):

    def __init__(self):
        pass

    def load_data(self, season: int) -> DataFrame:
        loader = PlayerLoader()
        df = loader.load_for_season(season)

        if df.empty:
            print("DF is empty")
        
        return df
    
    def _calc_rolling_window(self):
        pass

        
data = BattingFeatures().load_data(2021)
print(len(data))

db = get_database_manager()

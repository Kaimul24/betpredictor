"""
Handles computation of pitching stats features. Classifies each pitcher as starter/reliever.
"""

import pandas as pd
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader


class PitchingFeatures(BaseFeatures):

    def __init__(self, season: int):
        super().__init__(season)

    def load_data(self) -> DataFrame:
        loader = PlayerLoader()
        df = loader.load_for_season_pitcher(self.season)

        if df.empty:
            print("DF is empty")
        
        return df

    
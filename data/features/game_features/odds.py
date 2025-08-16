"""
Handles construction of odds features and converting to implied probabilities.
"""

from data.loaders.odds_loader import OddsLoader
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Optional
import pandas as pd

import numpy as np
class Odds(BaseFeatures):

    def __init__(self, data: DataFrame, season: int) -> None:
        super().__init__(season, data)
        
def main():
    odds_loader = OddsLoader()
    odds_data = odds_loader.load_for_season(2021)
    odds = Odds(odds_data, 2021)

if __name__ == "__main__":
    main()  
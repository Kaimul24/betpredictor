"""
Handles construction of odds features and converting to implied probabilities.
"""

from data.loaders.odds_loader import OddsLoader
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Optional

class Odds(BaseFeatures):

    def __init__(self, season: int) -> None:
        super().__init__(season)
    
    def load_data(self) -> DataFrame:
        return OddsLoader().load_for_season(self.season)
    
def main():
    odds = Odds(2021)
    odds_data = odds.load_data()
    print(len(odds_data))

if __name__ == "__main__":
    main()  
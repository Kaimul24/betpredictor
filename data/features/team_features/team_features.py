"""
Constructs features from team metrics such as record, 
record vs. team, strength of schedule, etc.
"""
from data.loaders.game_loader import GameLoader
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Optional

class TeamFeatures(BaseFeatures):

    def __init__(self, season: int):
        super().__init__(season)
    
    def load_data(self) -> DataFrame:
        return GameLoader().load_for_season(self.season)
         
    #number of games in season - used for rolling window (last k games)
    def calc_win_percentage(self, schedule: DataFrame, num_games: Optional[int] = None) -> float:
        """
        Calculates the win percentage of a team for the season and optionally, last num_games.
        
        Args:
            - schedule: DataFrame of schedule for a season
            - num_games: Optional, past number of games to calculate W%
        
        Returns:
            - win_percentage: Winning percentage as a decimal
        """
        pass
    

def main():
    team_feats = TeamFeatures(2021)
    team_data = team_feats.load_data()
    print(len(team_data))

if __name__ == "__main__":
    main()  
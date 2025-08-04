"""
Orchestrates the feature engineering process for all features. 
"""

import pandas as pd

from .game_features.context import GameContextFeatures
from .game_features.matchup import MatchupFeatures
from .game_features.odds import Odds
from .player_features.batting import BattingFeatures
from .player_features.fielding import FieldingFeatures
from .player_features.pitching import PitchingFeatures
from .player_features.war import WAR
from .team_features.schedule import ScheduleFeatures


class FeaturePipeline():

    def __init__(self, season: int):
        self.batting_features = BattingFeatures()
        self.pitching_features = PitchingFeatures()
        self.matchup_features = MatchupFeatures()
        self.fielding_features = FieldingFeatures()
        self.game_features = GameContextFeatures()
        self.schedule_features = ScheduleFeatures()
        self.war = WAR()
        self.odds = Odds()

    
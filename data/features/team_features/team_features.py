"""
Constructs features from team metrics game streaks/trends,
past strength of schedule, run diferential, home/away win percentage,
one run game win percentage, bullpen statuss
"""
from data.loaders.game_loader import GameLoader
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Optional
import pandas as pd

class TeamFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame):
        super().__init__(season, data)

        if 'team' not in self.data.index.names:
            raise RuntimeError("_transform_schedule() in feature_pipeline.py is meant to be called before any method in TeamFeatures is used.")
        
    def load_features(self) -> DataFrame:
        data = self.data.copy()
        
        win_pct = self.calc_win_pct()
        h2h = self.calc_h2h_pct()
        
        result = pd.merge(
            win_pct,
            h2h,
            left_index=True,
            right_index=True,
            how='inner' 
        )

        assert len(result) == len(win_pct) and len(result) == len(h2h)
        
        final_df = pd.merge(
            result,
            data[['game_id']],
            left_index=True,
            right_index=True,
            how="inner"
        )

        assert len(final_df) == len(result)
        return final_df

    def calc_win_pct(self) -> DataFrame:
        """Calculates the winning percentage of each team for each game in a season."""
        data = self.data.copy()
        data = data.sort_index(level=['team', 'game_date', 'dh', 'game_datetime'])

        g = data.groupby(level='team')

        previous_games = g.cumcount()
        previous_wins = g['is_winner'].cumsum().shift(1)
        win_pct = previous_wins / previous_games.replace(0, pd.NA)

        return DataFrame(win_pct, columns=['win_pct'])


    def calc_h2h_pct(self) -> DataFrame:
        """Calculates the Head-to-Head winning percentage for each team vs. their opponents in a season"""
        data = self.data.copy()
        data = data.sort_index(level=['team', 'game_date', 'dh', 'game_datetime'])

        g = data.groupby([data.index.get_level_values("team"), "opposing_team"])

        previous_games_vs_opp = g.cumcount()
        
        previous_wins_vs_opp = g['is_winner'].cumsum() - data['is_winner'].astype(int)
        win_pct_vs_opp = previous_wins_vs_opp / previous_games_vs_opp.replace(0, pd.NA)
        win_pct_vs_opp = win_pct_vs_opp.astype('Float64').fillna(0.0)

        return DataFrame(win_pct_vs_opp, columns=['win_pct_vs_opp'])
    

def main():
    from data.loaders.game_loader import GameLoader
    game_loader = GameLoader()
    data = game_loader.load_for_season(2021)

    from data.features.feature_pipeline import FeaturePipeline
    feat_pipe = FeaturePipeline(2021)

    transformed_data = feat_pipe._transform_schedule(data)

    team_feats = TeamFeatures(2021, transformed_data)
    team_feats = team_feats.load_features()
    print(team_feats)
    # win_pct = team_feats.calc_win_pct()
    # print(win_pct.tail())

    # h2h_pct = team_feats.calc_h2h_pct()
    # print(h2h_pct.tail(8))

if __name__ == "__main__":
    main()  

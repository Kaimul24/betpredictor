"""
Constructs features from team metrics game streaks/trends,
past strength of schedule, run diferential, home/away win percentage,
one run game win percentage, bullpen statuss
"""
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Optional
import logging
import pandas as pd
import numpy as np

class TeamFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame):
        super().__init__(season, data)

        if 'team' not in self.data.index.names:
            raise RuntimeError("_transform_schedule() in feature_pipeline.py is meant to be called before any method in TeamFeatures is used.")
        
    def load_features(self) -> DataFrame:
        
        win_pct = self.calc_rolling_win_pct()
        run_diff = self.calc_run_diff_metrics()
        one_run_games = self.calc_one_run_win_pct()
        

        result = pd.merge(
            win_pct,
            run_diff,
            left_index=True,
            right_index=True,
            how='inner'
        )

        result = pd.merge(
            result,
            one_run_games,
            left_index=True,
            right_index=True,
            how='inner'
        )

        return result
    
    def calc_rolling_win_pct(self) -> DataFrame:
        df = self.data.reset_index().copy()
        df.sort_values(['game_date', 'dh', 'team'], inplace=True)
        df['gp'] = 1

        prior_specs = {
            'win_pct_prior': ('is_winner', 'gp')
        }

        shrinkage_weights_cols = ['gp']

        ewm_cols = {
            'win_pct': ('is_winner', 'gp', 'win_pct_prior', 10, True)
        }
    
        preserve_cols = ['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team']
        
        team_grouping = df['team']
        
        result, _ = self.compute_rolling_stats(
            df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=team_grouping,
            halflives=(3, 8, 20)
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])
        result.drop(columns=['last_app_date'], inplace=True)

        return result
        

    def calc_h2h_pct(self) -> DataFrame:
        """Calculates the Head-to-Head winning percentage for each team vs. their opponents in a season"""
        df = self.data.reset_index().copy()
        df.sort_values(['game_date', 'dh', 'team'], inplace=True)
        
        df['gp_vs_opp'] = 1

        prior_specs = {
            'h2h_win_pct_prior': ('is_winner', 'gp_vs_opp')
        }
        
        shrinkage_weights_cols = ['gp_vs_opp']

        ewm_cols = {
            'h2h_win_pct': ('is_winner', 'gp_vs_opp', 'h2h_win_pct_prior', 5, True)
        }

        preserve_cols = ['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team']
        team_opp_grouping = pd.MultiIndex.from_arrays([df['team'], df['opposing_team']])
    
        result, _ = self.compute_rolling_stats(
            df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=team_opp_grouping,
            halflives=()
        )
        
        result = result.set_index(['game_id', 'team', 'game_date', 'dh', 'game_datetime'])
        
        h2h_cols = [col for col in result.columns if col.startswith('h2h_win_pct_season')]
        result = result[h2h_cols]
        
        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)
        
        return result
    
    def calc_one_run_win_pct(self) -> DataFrame:
        df = self.data.reset_index().copy()

        df.sort_values(['game_date', 'dh', 'team'], inplace=True)
        df['gp'] = 1

        # Create indicator for one-run games
        one_run_game = np.where(abs(df['team_score'] - df['opposing_team_score']) == 1, 1, 0)
        df['one_run_game'] = one_run_game
        
        # Create indicator for one-run wins (team wins AND it's a one-run game)
        df['one_run_win'] = np.where((df['is_winner'] == 1) & (df['one_run_game'] == 1), 1, 0)

        prior_specs = {
            'one_run_win_pct_prior': ('one_run_win', 'one_run_game')
        }

        shrinkage_weights_cols = ['one_run_game']

        ewm_cols = {
            'one_run_win_pct': ('one_run_win', 'one_run_game', 'one_run_win_pct_prior', 5, True)
        }

        preserve_cols = ['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team']
        
        team_grouping = df['team']
        
        result, _ = self.compute_rolling_stats(
            df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=team_grouping,
            halflives=(3, 8, 20)
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])
        
        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)

        return result
    
    def calc_run_diff_metrics(self) -> DataFrame:
        df = self.data.reset_index().copy()
        df.sort_values(['game_date', 'dh', 'team'], inplace=True)
        df['gp'] = 1

        rs = df['team_score'].astype(int)
        ra =  df['opposing_team_score'].astype(int)

        gamma = 1.83                        
        k = 10                              
        alpha = rs.mean() * k  

        RS_cum = BaseFeatures.cumsum_shift(rs, df['team'])
        RA_cum = BaseFeatures.cumsum_shift(ra, df['team'])
        num = (RS_cum + alpha) ** gamma
        den = num + (RA_cum + alpha) ** gamma
        df['pyth_expectation_season'] = (num / den).fillna(0.5)

        for hl in (3, 8, 20):
            rs_ewm = BaseFeatures.compute_ewm(rs, df['gp'], df['team'], hl, val_is_rate=False)
            ra_ewm = BaseFeatures.compute_ewm(ra, df['gp'], df['team'], hl, val_is_rate=False)
            num = (rs_ewm + alpha) ** gamma
            den = num + (ra_ewm + alpha) ** gamma
            df[f'pyth_expectation_ewm_h{hl}'] = (num / den).fillna(0.5)

        df['run_diff'] = rs - ra

        prior_specs = {
            "prior_run_diff": ("run_diff", "gp"),
        }

        shrinkage_weights_cols = ['gp']

        ewm_cols = {
            "run_diff": ("run_diff", "gp", "prior_run_diff", 10, False),
        }

        preserve_cols = ['game_id', 'game_date', 'dh', 'game_datetime', 'team', 
                        'opposing_team', 'pyth_expectation_season',
                        'pyth_expectation_ewm_h3', 'pyth_expectation_ewm_h8',
                        'pyth_expectation_ewm_h20']

        result, _ = BaseFeatures.compute_rolling_stats(
            data=df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=df['team'],
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])

        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)
        
        return result
    

def main():
    from src.data.loaders.game_loader import GameLoader
    game_loader = GameLoader()
    data = game_loader.load_for_season(2021)

    from src.data.features.feature_pipeline import FeaturePipeline
    feat_pipe = FeaturePipeline(2021, logger = logging.getLogger("team features"))

    transformed_data = feat_pipe._transform_schedule(data)

    team_feats = TeamFeatures(2021, transformed_data)
    team_feats = team_feats.load_features()
    print(team_feats.index.names)
    print(team_feats.columns)
    # win_pct = team_feats.calc_win_pct()
    # print(win_pct.tail())

    # h2h_pct = team_feats.calc_h2h_pct()
    # print(h2h_pct.tail(8))

if __name__ == "__main__":
    main()

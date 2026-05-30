"""
Constructs features from team metrics game streaks/trends,
past strength of schedule, run diferential, home/away win percentage,
one run game win percentage, bullpen statuss
"""
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import List
import logging
import pandas as pd
import numpy as np

class TeamFeatures(BaseFeatures):

    METRIC_BASES = ['win_pct', 'pyth_expectation', 'run_diff', 'one_run_win_pct']

    OUTPUT_INDEX = ['game_id', 'game_date', 'home_team', 'away_team', 'dh']

    def __init__(self, season: int, schedule_data: DataFrame, halflives: tuple[int, ...] = (3, 8, 20)):
        super().__init__(season, schedule_data)
        self.halflives = tuple(halflives)
        self.metric_cols = [
            col
            for base in self.METRIC_BASES
            for col in [f'{base}_season', *[f'{base}_ewm_h{hl}' for hl in self.halflives]]
        ]
        self.side_cols = self.metric_cols + ['team_gp']

    def load_features(self) -> DataFrame:
        team_log = self._build_team_game_log()

        win_pct = self._calc_rolling_win_pct(team_log)
        run_diff = self._calc_run_diff_metrics(team_log)
        one_run_games = self._calc_one_run_win_pct(team_log)

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

        result = self._team_games_played(result)
        result = self._collapse_to_game_level(result)

        return result

    def _build_team_game_log(self) -> DataFrame:
        """
        Builds the internal long per-team game log from the one row per game schedule.
        Each game becomes a home row and an away row so rolling team metrics can be
        computed across a team's full chronological history (home and away games).
        """
        df = self.data.reset_index().copy()
        common_cols = ['game_id', 'game_date', 'dh', 'game_datetime']

        away_df = df[common_cols].assign(
            is_home=False,
            team=df['away_team'],
            opposing_team=df['home_team'],
            team_score=df['away_score'],
            opposing_team_score=df['home_score'],
            is_winner=np.where(df['winning_team'] == df['away_team'], 1, 0),
        )

        home_df = df[common_cols].assign(
            is_home=True,
            team=df['home_team'],
            opposing_team=df['away_team'],
            team_score=df['home_score'],
            opposing_team_score=df['away_score'],
            is_winner=np.where(df['winning_team'] == df['home_team'], 1, 0),
        )

        log = pd.concat([away_df, home_df], ignore_index=True)
        log = log.sort_values(['game_date', 'dh', 'game_datetime', 'team']).reset_index(drop=True)

        return log

    def _team_games_played(self, df: DataFrame) -> DataFrame:

        df = df.copy().reset_index()
        df.sort_values(['game_date', 'dh', 'game_datetime', 'game_id'], inplace=True)

        df['team_gp'] = df.groupby('team').cumcount()
        df.set_index(keys=['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'], inplace=True)
        df.sort_index(level=['game_date', 'dh', 'game_datetime', 'team'], ascending=[True, True, True, True], inplace=True)

        return df

    def _collapse_to_game_level(self, df: DataFrame) -> DataFrame:
        """
        Collapses the team-indexed rolling stats into one row per game with
        home/away suffixed feature columns.
        """
        df = df.reset_index().copy()

        keys = ['game_id', 'game_date', 'dh', 'game_datetime']

        home_rows = df[df['is_home']].copy()
        away_rows = df[~df['is_home']].copy()

        home_rows = home_rows.rename(columns={col: f'home_{col}' for col in self.side_cols})
        away_rows = away_rows.rename(columns={col: f'away_{col}' for col in self.side_cols})

        home_rows = home_rows.rename(columns={'team': 'home_team'})
        away_rows = away_rows.rename(columns={'team': 'away_team'})

        home_keep = keys + ['home_team', 'is_winner'] + [f'home_{col}' for col in self.side_cols]
        away_keep = keys + ['away_team', 'is_winner'] + [f'away_{col}' for col in self.side_cols]

        merged = pd.merge(
            home_rows[home_keep],
            away_rows[away_keep],
            on=keys,
            how='inner',
            suffixes=("_home", "_away")
        )

        merged = merged.set_index(self.OUTPUT_INDEX)
        merged = merged.drop(columns=['game_datetime', 'is_winner_away'], errors='ignore')
        merged = merged.sort_index(level=['game_date', 'dh', ])
        
        return merged

    def _calc_rolling_win_pct(self, team_log: DataFrame) -> DataFrame:
        df = team_log.copy()
        df.sort_values(['game_date', 'dh', 'game_datetime', 'team'], inplace=True)
        df['gp'] = 1

        prior_specs = {
            'win_pct_prior': ('is_winner', 'gp')
        }

        shrinkage_weights_cols = ['gp']

        ewm_cols = {
            'win_pct': ('is_winner', 'gp', 'win_pct_prior', 10, True)
        }

        preserve_cols = ['game_id', 'game_date', 'dh', 'game_datetime', 'team', 'opposing_team', 'is_home', "is_winner"]
        team_grouping = df['team']

        result, _ = BaseFeatures.compute_rolling_stats(
            df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=team_grouping,
            halflives=self.halflives
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])

        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)

        return result

    # UNUSED
    def calc_h2h_pct(self, team_log: DataFrame) -> DataFrame:
        """Calculates the Head-to-Head winning percentage for each team vs. their opponents in a season"""
        df = team_log.copy()
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

        result, _ = BaseFeatures.compute_rolling_stats(
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

    def _calc_one_run_win_pct(self, team_log: DataFrame) -> DataFrame:
        df = team_log.copy()

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
            halflives=self.halflives
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])
        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)

        return result

    def _calc_run_diff_metrics(self, team_log: DataFrame) -> DataFrame:
        df = team_log.copy()
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

        for hl in self.halflives:
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
                        *[f'pyth_expectation_ewm_h{hl}' for hl in self.halflives]]

        result, _ = BaseFeatures.compute_rolling_stats(
            data=df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=preserve_cols,
            by=df['team'],
            halflives=self.halflives,
        )

        result = result.set_index(['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime'])

        if 'last_app_date' in result.columns:
            result.drop(columns=['last_app_date'], inplace=True)
        
        return result
    

def main():
    from src.data.loaders.game_loader import GameLoader
    game_loader = GameLoader()
    data = game_loader.load_for_season(2021)

    team_feats = TeamFeatures(2021, data)
    team_feats = team_feats.load_features()
    print(team_feats.index.names)
    print(team_feats.columns)

if __name__ == "__main__":
    main()

import pytest
import pandas as pd

from src.data.features.team_features.team_features import TeamFeatures
from tests.conftest import assert_dataframe_schema, assert_dataframe_not_empty


class TestTeamFeatures:
    """Test suite for the game-level (one row per game) home/away TeamFeatures contract."""

    PROB_BASES = ['win_pct', 'pyth_expectation', 'one_run_win_pct']
    METRIC_SUFFIXES = ['season', 'ewm_h3', 'ewm_h8', 'ewm_h20']

    @pytest.fixture
    def sample_schedule_df(self):
        columns = [
            'game_id', 'game_date', 'game_datetime', 'dh',
            'home_team', 'away_team', 'home_score', 'away_score', 'winning_team',
        ]
        games = [
            ('g1', '2021-04-01', '2021-04-01 19:05', 0, 'SFG', 'LAD', 3, 2, 'SFG'),
            ('g2', '2021-04-02', '2021-04-02 19:05', 0, 'LAD', 'SFG', 4, 1, 'LAD'),
            ('g3', '2021-04-03', '2021-04-03 19:05', 0, 'SFG', 'NYM', 5, 2, 'SFG'),
            ('g4', '2021-04-04', '2021-04-04 19:05', 0, 'NYM', 'SFG', 3, 2, 'NYM'),
            ('g5', '2021-04-05', '2021-04-05 19:05', 0, 'LAD', 'NYM', 6, 1, 'LAD'),
        ]

        df = pd.DataFrame(games, columns=columns)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])
        return df

    @pytest.fixture
    def team_features(self, sample_schedule_df):
        return TeamFeatures(2021, sample_schedule_df)

    @staticmethod
    def _side_columns():
        cols = []
        for base in TeamFeatures.SIDE_COLS:
            cols.append(f'home_{base}')
            cols.append(f'away_{base}')
        return cols

    def test_load_features_is_one_row_per_game(self, team_features):
        result = team_features.load_features()

        assert_dataframe_not_empty(result)
        assert len(result) == 5
        assert result.index.names == ['game_id', 'game_date', 'home_team', 'away_team', 'dh']
        assert not result.index.has_duplicates

    def test_load_features_has_home_away_columns(self, team_features):
        result = team_features.load_features()

        assert_dataframe_schema(result, self._side_columns())

        # No leftover team-perspective helper columns remain.
        for leftover in ['team', 'opposing_team', 'is_home', 'game_datetime', 'opposing_team_gp']:
            assert leftover not in result.columns
            assert leftover not in result.index.names

    def test_first_game_priors(self, team_features):
        result = team_features.load_features()

        # g1 is the first game for both SFG (home) and LAD (away).
        g1 = result.xs('g1', level='game_id').iloc[0]

        assert g1['home_win_pct_season'] == pytest.approx(0.5)
        assert g1['away_win_pct_season'] == pytest.approx(0.5)
        assert g1['home_pyth_expectation_season'] == pytest.approx(0.5)
        assert g1['away_pyth_expectation_season'] == pytest.approx(0.5)
        assert g1['home_run_diff_season'] == pytest.approx(0.0)
        assert g1['away_run_diff_season'] == pytest.approx(0.0)

    def test_probability_columns_bounded(self, team_features):
        result = team_features.load_features()

        prob_cols = [
            f'{side}_{base}_{suffix}'
            for base in self.PROB_BASES
            for suffix in self.METRIC_SUFFIXES
            for side in ('home', 'away')
        ]

        for column in prob_cols:
            non_na = result[column].dropna()
            assert ((non_na >= 0.0) & (non_na <= 1.0)).all()

    def test_team_games_played_counts(self, team_features):
        result = team_features.load_features().reset_index()

        def gp(game_id, side):
            row = result[result['game_id'] == game_id].iloc[0]
            return row[f'{side}_team_gp']

        # g1: both teams in their first game.
        assert gp('g1', 'home') == 0  # SFG
        assert gp('g1', 'away') == 0  # LAD
        # g2: LAD (home) and SFG (away) have each played one prior game.
        assert gp('g2', 'home') == 1  # LAD
        assert gp('g2', 'away') == 1  # SFG
        # g3: SFG (home) has played g1, g2; NYM (away) is in its first game.
        assert gp('g3', 'home') == 2  # SFG
        assert gp('g3', 'away') == 0  # NYM
        # g4: NYM (home) played g3; SFG (away) played g1, g2, g3.
        assert gp('g4', 'home') == 1  # NYM
        assert gp('g4', 'away') == 3  # SFG
        # g5: LAD (home) played g1, g2; NYM (away) played g3, g4.
        assert gp('g5', 'home') == 2  # LAD
        assert gp('g5', 'away') == 2  # NYM

    @pytest.fixture
    def empty_schedule_data(self):
        return pd.DataFrame({
            'game_id': pd.Series(dtype='object'),
            'game_date': pd.Series(dtype='datetime64[ns]'),
            'game_datetime': pd.Series(dtype='datetime64[ns]'),
            'dh': pd.Series(dtype='int64'),
            'home_team': pd.Series(dtype='object'),
            'away_team': pd.Series(dtype='object'),
            'home_score': pd.Series(dtype='int64'),
            'away_score': pd.Series(dtype='int64'),
            'winning_team': pd.Series(dtype='object'),
        })

    def test_empty_data_handling(self, empty_schedule_data):
        team_features = TeamFeatures(2021, empty_schedule_data)

        result = team_features.load_features()

        assert len(result) == 0
        assert result.index.names == ['game_id', 'game_date', 'home_team', 'away_team', 'dh']
        assert_dataframe_schema(result, self._side_columns())

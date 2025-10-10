import pytest
import pandas as pd

from src.data.features.team_features.team_features import TeamFeatures
from tests.conftest import assert_dataframe_schema, assert_dataframe_not_empty


class TestTeamFeatures:
    """Test suite for TeamFeatures aligned with current feature calculations."""

    @pytest.fixture
    def sample_schedule_df(self):
        columns = [
            'game_id', 'game_date', 'dh', 'team', 'game_datetime',
            'opposing_team', 'is_winner', 'team_score', 'opposing_team_score'
        ]
        games = [
            ('g1', '2021-04-01', 0, 'SFG', '2021-04-01 19:05', 'LAD', 1, 3, 2),
            ('g1', '2021-04-01', 0, 'LAD', '2021-04-01 19:05', 'SFG', 0, 2, 3),
            ('g2', '2021-04-02', 0, 'SFG', '2021-04-02 19:05', 'LAD', 0, 1, 4),
            ('g2', '2021-04-02', 0, 'LAD', '2021-04-02 19:05', 'SFG', 1, 4, 1),
            ('g3', '2021-04-03', 0, 'SFG', '2021-04-03 19:05', 'NYM', 1, 5, 2),
            ('g3', '2021-04-03', 0, 'NYM', '2021-04-03 19:05', 'SFG', 0, 2, 5),
            ('g4', '2021-04-04', 0, 'SFG', '2021-04-04 19:05', 'NYM', 0, 2, 3),
            ('g4', '2021-04-04', 0, 'NYM', '2021-04-04 19:05', 'SFG', 1, 3, 2),
            ('g5', '2021-04-05', 0, 'LAD', '2021-04-05 19:05', 'NYM', 1, 6, 1),
            ('g5', '2021-04-05', 0, 'NYM', '2021-04-05 19:05', 'LAD', 0, 1, 6),
        ]

        df = pd.DataFrame(games, columns=columns)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])

        df = df.set_index(['game_date', 'dh', 'team', 'game_datetime'])
        return df

    @pytest.fixture
    def team_features(self, sample_schedule_df):
        return TeamFeatures(2021, sample_schedule_df)

    def test_calc_rolling_win_pct_structure(self, team_features):
        result = team_features.calc_rolling_win_pct()

        assert_dataframe_not_empty(result)

        expected_columns = ['win_pct_season', 'win_pct_ewm_h3', 'win_pct_ewm_h8', 'win_pct_ewm_h20']
        assert_dataframe_schema(result, expected_columns)
        assert len(result) == len(team_features.data)
        assert result.index.names == ['game_id', 'team', 'opposing_team', 'game_date', 'dh', 'game_datetime']

        for team, group in result.groupby(level='team'):
            ordered = group.sort_index(level=['game_date', 'dh', 'game_datetime'])
            first_row = ordered.iloc[0]
            assert first_row['win_pct_season'] == pytest.approx(0.5)

        for column in expected_columns:
            non_na = result[column].dropna()
            assert ((non_na >= 0.0) & (non_na <= 1.0)).all()

    def test_calc_one_run_win_pct(self, team_features):
        result = team_features.calc_one_run_win_pct()

        assert_dataframe_not_empty(result)

        expected_columns = [
            'one_run_win_pct_season',
            'one_run_win_pct_ewm_h3',
            'one_run_win_pct_ewm_h8',
            'one_run_win_pct_ewm_h20',
        ]
        assert_dataframe_schema(result, expected_columns)

        for team, group in result.groupby(level='team'):
            ordered = group.sort_index(level=['game_date', 'dh', 'game_datetime'])
            first_row = ordered.iloc[0]
            assert first_row['one_run_win_pct_season'] == pytest.approx(0.5)

        for column in expected_columns:
            non_na = result[column].dropna()
            assert ((non_na >= 0.0) & (non_na <= 1.0)).all()

    def test_calc_run_diff_metrics(self, team_features):
        result = team_features.calc_run_diff_metrics()

        assert_dataframe_not_empty(result)

        expected_columns = [
            'pyth_expectation_season',
            'pyth_expectation_ewm_h3',
            'pyth_expectation_ewm_h8',
            'pyth_expectation_ewm_h20',
            'run_diff_season',
            'run_diff_ewm_h3',
            'run_diff_ewm_h8',
            'run_diff_ewm_h20',
        ]
        assert_dataframe_schema(result, expected_columns)

        first_by_team = result.groupby(level='team').first()
        for team in first_by_team.index:
            assert first_by_team.loc[team, 'pyth_expectation_season'] == pytest.approx(0.5)
            assert first_by_team.loc[team, 'run_diff_season'] == pytest.approx(0.0)

        for column in [c for c in expected_columns if c.startswith('pyth_expectation')]:
            non_na = result[column].dropna()
            assert ((non_na >= 0.0) & (non_na <= 1.0)).all()

    # def test_calc_h2h_pct(self, team_features):
    #     result = team_features.calc_h2h_pct()

    #     assert_dataframe_not_empty(result)
    #     assert_dataframe_schema(result, ['h2h_win_pct_season'])
    #     assert result.index.names == ['game_id', 'team', 'game_date', 'dh', 'game_datetime']

    #     sfg_vs_lad = result.xs(('SFG', 'LAD'), level=('team', 'opposing_team')).sort_index(
    #         level=['game_date', 'dh', 'game_datetime']
    #     )
    #     assert sfg_vs_lad.iloc[0]['h2h_win_pct_season'] == pytest.approx(0.5)
    #     assert sfg_vs_lad.iloc[-1]['h2h_win_pct_season'] > 0.5

    def test_load_features_combines_all_metrics(self, team_features):
        result = team_features.load_features()

        assert_dataframe_not_empty(result)

        expected_columns = [
            'win_pct_season',
            'win_pct_ewm_h3',
            'win_pct_ewm_h8',
            'win_pct_ewm_h20',
            'pyth_expectation_season',
            'pyth_expectation_ewm_h3',
            'pyth_expectation_ewm_h8',
            'pyth_expectation_ewm_h20',
            'run_diff_season',
            'run_diff_ewm_h3',
            'run_diff_ewm_h8',
            'run_diff_ewm_h20',
            'one_run_win_pct_season',
            'one_run_win_pct_ewm_h3',
            'one_run_win_pct_ewm_h8',
            'one_run_win_pct_ewm_h20',
            'team_gp',
            'opposing_team_gp',
        ]
        assert_dataframe_schema(result, expected_columns)

        ordered = result.reset_index().sort_values(
            ['game_date', 'dh', 'game_datetime', 'game_id', 'team']
        )

        for team, group in ordered.groupby('team'):
            assert list(group['team_gp']) == list(range(len(group)))

        for opponent, group in ordered.groupby('opposing_team'):
            assert list(group['opposing_team_gp']) == list(range(len(group)))

    def test_initialization_validation(self, sample_schedule_df):
        invalid_data = sample_schedule_df.reset_index('team')

        with pytest.raises(RuntimeError, match="_transform_schedule\(\).+TeamFeatures"):
            TeamFeatures(2021, invalid_data)

    @pytest.fixture
    def empty_schedule_data(self):
        df = pd.DataFrame({
            'game_id': pd.Series(dtype='object'),
            'opposing_team': pd.Series(dtype='object'),
            'is_winner': pd.Series(dtype='int64'),
            'team_score': pd.Series(dtype='int64'),
            'opposing_team_score': pd.Series(dtype='int64'),
        })

        df.index = pd.MultiIndex.from_arrays(
            [
                pd.Index([], dtype='datetime64[ns]'),
                pd.Index([], dtype='int64'),
                pd.Index([], dtype='object'),
                pd.Index([], dtype='datetime64[ns]'),
            ],
            names=['game_date', 'dh', 'team', 'game_datetime']
        )

        return df

    def test_empty_data_handling(self, empty_schedule_data):
        team_features = TeamFeatures(2021, empty_schedule_data)

        rolling = team_features.calc_rolling_win_pct()
        run_diff = team_features.calc_run_diff_metrics()
        one_run = team_features.calc_one_run_win_pct()
        merged = team_features.load_features()

        assert len(rolling) == 0
        assert len(run_diff) == 0
        assert len(one_run) == 0
        assert len(merged) == 0

        assert_dataframe_schema(rolling, ['win_pct_season', 'win_pct_ewm_h3', 'win_pct_ewm_h8', 'win_pct_ewm_h20'])
        assert_dataframe_schema(run_diff, [
            'pyth_expectation_season', 'pyth_expectation_ewm_h3', 'pyth_expectation_ewm_h8',
            'pyth_expectation_ewm_h20', 'run_diff_season', 'run_diff_ewm_h3', 'run_diff_ewm_h8',
            'run_diff_ewm_h20'
        ])
        assert_dataframe_schema(one_run, [
            'one_run_win_pct_season', 'one_run_win_pct_ewm_h3', 'one_run_win_pct_ewm_h8',
            'one_run_win_pct_ewm_h20'
        ])
        assert_dataframe_schema(merged, [
            'win_pct_season', 'win_pct_ewm_h3', 'win_pct_ewm_h8', 'win_pct_ewm_h20',
            'pyth_expectation_season', 'pyth_expectation_ewm_h3', 'pyth_expectation_ewm_h8',
            'pyth_expectation_ewm_h20', 'run_diff_season', 'run_diff_ewm_h3', 'run_diff_ewm_h8',
            'run_diff_ewm_h20', 'one_run_win_pct_season', 'one_run_win_pct_ewm_h3',
            'one_run_win_pct_ewm_h8', 'one_run_win_pct_ewm_h20', 'team_gp', 'opposing_team_gp'
        ])

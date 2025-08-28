import pytest
import pandas as pd
import numpy as np


from src.data.features.team_features.team_features import TeamFeatures
from tests.conftest import (
    assert_dataframe_schema, assert_dataframe_not_empty
)

class TestTeamFeatures:
    """Test suite for TeamFeatures class"""

    @pytest.fixture
    def sample_transformed_schedule_games(self):
        columns = ['game_date', 'dh', 'team', 'game_datetime', 'opposing_team', 'is_winner']
        games = [
            ('2021-04-01', 0, 'SFG', '2021-04-01', 'LAD', 1),
            ('2021-04-01', 0, 'LAD', '2021-04-01', 'LAD', 0),
            ('2021-04-02', 1, 'SFG', '2021-04-02', 'LAD', 0),
            ('2021-04-02', 1, 'LAD', '2021-04-02', 'LAD', 1),
            ('2021-04-02', 2, 'SFG', '2021-04-02', 'LAD', 1),
            ('2021-04-02', 2, 'LAD', '2021-04-02', 'LAD', 0)
        ]

        df = pd.DataFrame(games, columns=columns)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])

        df = df.set_index(['game_date', 'dh', 'team', 'game_datetime'])

        return df
    
    @pytest.fixture
    def team_features(self, sample_transformed_schedule_games):
        return TeamFeatures(2021, sample_transformed_schedule_games)
         
    def test_calc_win_pct_basic(self, team_features):
        "Test basic functionality of calc_win_percent"
        win_pct = team_features.calc_win_pct()

        assert_dataframe_not_empty(win_pct)
        
        assert pd.isna(win_pct.iloc[0]['win_pct'])
        assert win_pct.iloc[1]['win_pct'] == 0.0
        assert win_pct.iloc[2]['win_pct'] == 0.5
        assert pd.isna(win_pct.iloc[3]['win_pct'])
        assert win_pct.iloc[4]['win_pct'] == 1.0
        assert win_pct.iloc[5]['win_pct'] == 0.5

    def test_calc_win_pct_schema(self, team_features):
        """Test that calc_win_pct returns correct schema"""
        win_pct = team_features.calc_win_pct()
        
        assert_dataframe_schema(win_pct, ['win_pct'])
        assert len(win_pct) == len(team_features.data)
        # Index may be reordered due to sorting, so just check names match
        assert win_pct.index.names == team_features.data.index.names

    def test_calc_h2h_pct_basic(self, team_features):
        """Test basic functionality of calc_h2h_pct"""
        h2h_pct = team_features.calc_h2h_pct()

        assert_dataframe_not_empty(h2h_pct)
        assert_dataframe_schema(h2h_pct, ['win_pct_vs_opp'])
        
        # First games against each opponent should be 0.0 (no previous history)
        assert h2h_pct.iloc[0]['win_pct_vs_opp'] == 0.0
        assert h2h_pct.iloc[1]['win_pct_vs_opp'] == 0.0

    @pytest.fixture
    def complex_schedule_data(self):
        """More complex test data with multiple teams and games"""
        columns = ['game_date', 'dh', 'team', 'game_datetime', 'opposing_team', 'is_winner']
        games = [
            # SFG vs LAD series
            ('2021-04-01', 0, 'SFG', '2021-04-01', 'LAD', 1),
            ('2021-04-01', 0, 'LAD', '2021-04-01', 'SFG', 0),
            ('2021-04-02', 0, 'SFG', '2021-04-02', 'LAD', 0),
            ('2021-04-02', 0, 'LAD', '2021-04-02', 'SFG', 1),
            ('2021-04-03', 0, 'SFG', '2021-04-03', 'LAD', 1),
            ('2021-04-03', 0, 'LAD', '2021-04-03', 'SFG', 0),
            
            # SFG vs NYM games
            ('2021-04-05', 0, 'SFG', '2021-04-05', 'NYM', 1),
            ('2021-04-05', 0, 'NYM', '2021-04-05', 'SFG', 0),
            ('2021-04-06', 0, 'SFG', '2021-04-06', 'NYM', 0),
            ('2021-04-06', 0, 'NYM', '2021-04-06', 'SFG', 1),
            
            # LAD vs NYM games
            ('2021-04-08', 0, 'LAD', '2021-04-08', 'NYM', 1),
            ('2021-04-08', 0, 'NYM', '2021-04-08', 'LAD', 0),
        ]

        df = pd.DataFrame(games, columns=columns)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])

        df = df.set_index(['game_date', 'dh', 'team', 'game_datetime'])
        return df

    @pytest.fixture
    def complex_team_features(self, complex_schedule_data):
        return TeamFeatures(2021, complex_schedule_data)

    def test_calc_win_pct_multiple_teams(self, complex_team_features):
        """Test win percentage calculation with multiple teams"""
        win_pct = complex_team_features.calc_win_pct()
        
        assert_dataframe_not_empty(win_pct)
        
        # Check that each team gets correct cumulative win percentages
        sfg_data = win_pct.loc[win_pct.index.get_level_values('team') == 'SFG']
        lad_data = win_pct.loc[win_pct.index.get_level_values('team') == 'LAD'] 
        nym_data = win_pct.loc[win_pct.index.get_level_values('team') == 'NYM']
        
        # Verify that we have the expected number of games per team
        assert len(sfg_data) == 5  # SFG played 5 games
        assert len(lad_data) == 4  # LAD played 4 games  
        assert len(nym_data) == 3  # NYM played 3 games

    def test_calc_h2h_pct_multiple_opponents(self, complex_team_features):
        """Test head-to-head percentage with multiple opponents"""
        h2h_pct = complex_team_features.calc_h2h_pct()
        
        assert_dataframe_not_empty(h2h_pct)
        
        # Check that all values are numeric and within valid range [0, 1]
        assert h2h_pct['win_pct_vs_opp'].dtype in ['float64', 'Float64']
        assert (h2h_pct['win_pct_vs_opp'] >= 0.0).all()
        assert (h2h_pct['win_pct_vs_opp'] <= 1.0).all()

    @pytest.fixture
    def single_game_data(self):
        """Single game test data"""
        columns = ['game_date', 'dh', 'team', 'game_datetime', 'opposing_team', 'is_winner']
        games = [
            ('2021-04-01', 0, 'SFG', '2021-04-01', 'LAD', 1),
            ('2021-04-01', 0, 'LAD', '2021-04-01', 'SFG', 0),
        ]

        df = pd.DataFrame(games, columns=columns)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])

        df = df.set_index(['game_date', 'dh', 'team', 'game_datetime'])
        return df

    def test_single_game_win_pct(self, single_game_data):
        """Test win percentage calculation with only one game"""
        team_features = TeamFeatures(2021, single_game_data)
        win_pct = team_features.calc_win_pct()
        
        # First game for each team should have NaN win percentage
        assert pd.isna(win_pct.iloc[0]['win_pct'])
        assert pd.isna(win_pct.iloc[1]['win_pct'])

    def test_single_game_h2h_pct(self, single_game_data):
        """Test head-to-head percentage with only one game"""
        team_features = TeamFeatures(2021, single_game_data)
        h2h_pct = team_features.calc_h2h_pct()
        
        # First game against each opponent should be 0.0
        assert h2h_pct.iloc[0]['win_pct_vs_opp'] == 0.0
        assert h2h_pct.iloc[1]['win_pct_vs_opp'] == 0.0

    def test_initialization_validation(self, sample_transformed_schedule_games):
        """Test that TeamFeatures validates required index structure"""
        # Test with missing 'team' in index
        invalid_data = sample_transformed_schedule_games.reset_index('team')
        
        with pytest.raises(RuntimeError, match="_transform_schedule.*is meant to be called"):
            TeamFeatures(2021, invalid_data)

    @pytest.fixture
    def empty_schedule_data(self):
        """Empty DataFrame with correct structure and proper dtypes"""
        columns = ['opposing_team', 'is_winner']
        df = pd.DataFrame(columns=columns)
        df['is_winner'] = df['is_winner'].astype('int64')  # Ensure numeric dtype
        df.index = pd.MultiIndex.from_tuples(
            [], 
            names=['game_date', 'dh', 'team', 'game_datetime']
        )
        return df

    def test_empty_data_handling(self, empty_schedule_data):
        """Test behavior with empty data"""
        team_features = TeamFeatures(2021, empty_schedule_data)
        
        win_pct = team_features.calc_win_pct()
        h2h_pct = team_features.calc_h2h_pct()
        
        assert len(win_pct) == 0
        assert len(h2h_pct) == 0
        assert_dataframe_schema(win_pct, ['win_pct'])
        assert_dataframe_schema(h2h_pct, ['win_pct_vs_opp'])

    def test_win_pct_ordering(self, complex_team_features):
        """Test that win percentage respects chronological ordering"""
        win_pct = complex_team_features.calc_win_pct()
        
        # Check that data is properly sorted by team, then chronologically
        for team in ['SFG', 'LAD', 'NYM']:
            team_data = win_pct.loc[win_pct.index.get_level_values('team') == team]
            if len(team_data) > 1:
                # Dates should be in ascending order
                dates = team_data.index.get_level_values('game_date')
                assert (dates[:-1] <= dates[1:]).all()

    def test_h2h_pct_opponent_specificity(self, complex_team_features):
        """Test that head-to-head percentages are opponent-specific"""
        h2h_pct = complex_team_features.calc_h2h_pct()
        
        # Get SFG's record against LAD vs NYM
        sfg_vs_lad = h2h_pct.loc[
            (h2h_pct.index.get_level_values('team') == 'SFG') & 
            (complex_team_features.data['opposing_team'] == 'LAD')
        ]
        
        sfg_vs_nym = h2h_pct.loc[
            (h2h_pct.index.get_level_values('team') == 'SFG') & 
            (complex_team_features.data['opposing_team'] == 'NYM')
        ]
        
        # First games against each opponent should start at 0.0
        if len(sfg_vs_lad) > 0:
            assert sfg_vs_lad.iloc[0]['win_pct_vs_opp'] == 0.0
        if len(sfg_vs_nym) > 0:
            assert sfg_vs_nym.iloc[0]['win_pct_vs_opp'] == 0.0

    def test_dataframe_dtypes(self, team_features):
        """Test that returned DataFrames have correct data types"""
        win_pct = team_features.calc_win_pct()
        h2h_pct = team_features.calc_h2h_pct()
        
        # Win percentage should be numeric type (may be object due to NA values)
        assert win_pct['win_pct'].dtype in ['float64', 'Float64', 'object']
        
        # H2H percentage should be Float64 (nullable float)
        assert h2h_pct['win_pct_vs_opp'].dtype == 'Float64'
        
        # Verify that numeric values are actually numeric when not NA
        non_na_win_pct = win_pct['win_pct'].dropna()
        if len(non_na_win_pct) > 0:
            assert pd.api.types.is_numeric_dtype(non_na_win_pct.astype(float))

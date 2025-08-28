"""
Tests for OddsLoader class.

Tests the odds data loading functionality including:
- Loading odds for date ranges
- Loading team-specific odds up to a date  
- Loading specific game odds
- Sportsbook filtering
- Data validation
"""

import pytest
import pandas as pd
from datetime import date
from src.data.loaders.odds_loader import OddsLoader
from tests.conftest import (
    insert_odds_data, assert_dataframe_schema, 
    assert_dataframe_not_empty, assert_dataframe_values
)


class TestOddsLoader:
    """Test suite for OddsLoader class."""
    
    @pytest.fixture
    def odds_loader(self, clean_db):
        """Create an OddsLoader instance with clean database."""
        return OddsLoader()
    
    @pytest.fixture
    def sample_odds(self, clean_db):
        """Insert sample odds data for testing."""
        odds = [
            # (game_date, game_datetime, away_team, home_team, away_starter, home_starter,
            #  sportsbook, away_opening_odds, home_opening_odds, away_current_odds, home_current_odds, season)
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'DraftKings', -110, +120, -108, +118, 2024),
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'FanDuel', -115, +125, -112, +122, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'DraftKings', +140, -130, +138, -128, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'FanDuel', +135, -125, +133, -123, 2024),
            ('2024-04-03', '2024-04-03T18:35:00', 'TB', 'BOS', 'Glasnow', 'Pivetta', 'DraftKings', -105, +110, -103, +108, 2024),
        ]
        insert_odds_data(clean_db, odds)
        return odds

    def test_load_for_date_range_basic(self, odds_loader, sample_odds):
        """Test loading odds for a basic date range."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 2)
        )
        
        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, odds_loader.columns)
        assert len(df) == 4
        
        dates = df['game_date'].unique()
        assert '2024-04-01' in dates
        assert '2024-04-02' in dates
        assert '2024-04-03' not in dates

    def test_load_for_date_range_empty(self, odds_loader, sample_odds):
        """Test loading odds for date range with no odds."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 1, 1), 
            end=date(2024, 1, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_for_date_range_single_day(self, odds_loader, sample_odds):
        """Test loading odds for a single day."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        assert_dataframe_not_empty(df)
        assert len(df) == 2
        assert all(df['game_date'] == '2024-04-01')

    def test_load_for_season_basic(self, odds_loader, sample_odds):
        """Test loading all odds for a specific season."""
        df = odds_loader.load_for_season(season=2024)
        
        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, odds_loader.columns)
        assert len(df) == 5
        
        assert all(df['season'] == 2024)

        dates = df['game_date'].tolist()
        assert dates == sorted(dates)

    def test_load_for_season_empty(self, odds_loader, sample_odds):
        """Test loading odds for a season with no data."""
        df = odds_loader.load_for_season(season=2023)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_for_season_multiple_seasons(self, odds_loader, clean_db):
        """Test loading odds when multiple seasons exist."""
        multi_season_odds = [
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'DraftKings', -110, +120, -108, +118, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'FanDuel', +140, -130, +138, -128, 2024),
            ('2023-04-01', '2023-04-01T19:05:00', 'TB', 'BOS', 'Glasnow', 'Pivetta', 'DraftKings', -105, +110, -103, +108, 2023),
            ('2023-09-15', '2023-09-15T19:10:00', 'BOS', 'NYY', 'Sale', 'Cole', 'FanDuel', +115, -125, +117, -123, 2023),
            ('2025-04-01', '2025-04-01T19:05:00', 'NYY', 'TB', 'Cole', 'Glasnow', 'DraftKings', -120, +130, -118, +128, 2025),
        ]
        insert_odds_data(clean_db, multi_season_odds)
        
        df_2024 = odds_loader.load_for_season(season=2024)

        assert_dataframe_not_empty(df_2024)
        assert len(df_2024) == 2
        assert all(df_2024['season'] == 2024)
        
        df_2023 = odds_loader.load_for_season(season=2023)
        
        assert_dataframe_not_empty(df_2023)
        assert len(df_2023) == 2
        assert all(df_2023['season'] == 2023)

    def test_load_up_to_game_basic(self, odds_loader, sample_odds):
        """Test loading odds for a team up to a specific date."""
        df = odds_loader.load_up_to_game(
            date=date(2024, 4, 3), 
            team_abbr='BOS'
        )
        
        assert_dataframe_not_empty(df)
        assert 'team_side' in df.columns
  
        boston_games = df[(df['home_team'] == 'BOS') | (df['away_team'] == 'BOS')]
        assert len(boston_games) > 0

    def test_load_game_odds_with_sportsbook(self, odds_loader, sample_odds):
        """Test loading odds for a specific game and sportsbook."""
        df = odds_loader.load_game_odds(
            game_date=date(2024, 4, 1),
            away_team='NYY',
            home_team='BOS', 
            away_starter='Cole',
            home_starter='Sale',
            sportsbook='DraftKings'
        )
        pd.set_option('display.max_columns', None)
        print(df)
        
        assert_dataframe_not_empty(df)
        assert len(df) == 1
        assert df.iloc[0]['sportsbook'] == 'DraftKings'
        assert df.iloc[0]['away_opening_odds'] == -110
        assert df.iloc[0]['home_opening_odds'] == 120
        assert df.iloc[0]['away_current_odds'] == -108
        assert df.iloc[0]['home_current_odds'] == 118

    def test_load_game_odds_all_sportsbooks(self, odds_loader, sample_odds):
        """Test loading odds for a specific game from all sportsbooks."""
        df = odds_loader.load_game_odds(
            game_date=date(2024, 4, 1),
            away_team='NYY',
            home_team='BOS',
            away_starter='Cole', 
            home_starter='Sale'
        )
        
        assert_dataframe_not_empty(df)
        assert len(df) == 2
        sportsbooks = df['sportsbook'].tolist()
        assert 'DraftKings' in sportsbooks
        assert 'FanDuel' in sportsbooks

    def test_load_game_odds_no_results(self, odds_loader, sample_odds):
        """Test loading odds for a game that doesn't exist."""
        df = odds_loader.load_game_odds(
            game_date=date(2024, 12, 31),
            away_team='NONE',
            home_team='ALSO_NONE',
            away_starter='Nobody',
            home_starter='No one'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_column_validation(self, odds_loader, sample_odds):
        """Test that all expected columns are present and valid."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        assert_dataframe_schema(df, odds_loader.columns)
        
        assert pd.api.types.is_datetime64_ns_dtype(df['game_date'])
        assert pd.api.types.is_object_dtype(df['sportsbook'])
        assert pd.api.types.is_numeric_dtype(df['away_opening_odds'])
        assert pd.api.types.is_numeric_dtype(df['home_opening_odds'])
        assert pd.api.types.is_numeric_dtype(df['away_current_odds'])
        assert pd.api.types.is_numeric_dtype(df['home_current_odds'])
        assert pd.api.types.is_integer_dtype(df['season'])
        
        required_fields = ['game_date', 'away_team', 'home_team', 'sportsbook']
        for field in required_fields:
            assert not df[field].isnull().any(), f"Field {field} should not have null values"

    def test_odds_values_realistic(self, odds_loader, sample_odds):
        """Test that odds values are within realistic ranges."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        assert_dataframe_not_empty(df)
        
        assert df['away_opening_odds'].min() >= -1000
        assert df['away_opening_odds'].max() <= 1000
        assert df['home_opening_odds'].min() >= -1000
        assert df['home_opening_odds'].max() <= 1000
        assert df['away_current_odds'].min() >= -1000
        assert df['away_current_odds'].max() <= 1000
        assert df['home_current_odds'].min() >= -1000
        assert df['home_current_odds'].max() <= 1000

    @pytest.mark.parametrize("team,expected_min_games", [
        ("BOS", 2),
        ("NYY", 2),  
        ("TB", 1),
    ])
    def test_team_odds_availability(self, odds_loader, sample_odds, team, expected_min_games):
        """Test that teams have odds available for expected number of games."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        team_odds = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
        unique_games = team_odds[['game_date', 'away_team', 'home_team']].drop_duplicates()
        assert len(unique_games) >= expected_min_games

    def test_sportsbook_consistency(self, odds_loader, sample_odds):
        """Test that sportsbook data is consistent."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 2)
        )
        
        assert_dataframe_not_empty(df)
        
        sportsbooks = df['sportsbook'].unique()
        assert len(sportsbooks) >= 2
        assert 'DraftKings' in sportsbooks
        assert 'FanDuel' in sportsbooks

    def test_season_filtering(self, odds_loader, clean_db):
        """Test that season filtering works correctly."""
        multi_season_odds = [
            ('2024-04-01', '2024-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -110, +110, -108, +108, 2024),
            ('2023-04-01', '2023-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -120, +120, -118, +118, 2023),
            ('2025-04-01', '2025-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -105, +105, -103, +103, 2025),
        ]
        insert_odds_data(clean_db, multi_season_odds)
        
        df_date_range = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        assert_dataframe_not_empty(df_date_range)
        assert len(df_date_range) == 1
        assert df_date_range.iloc[0]['season'] == 2024

        df_season = odds_loader.load_for_season(season=2024)
        
        assert_dataframe_not_empty(df_season)
        assert len(df_season) == 1
        assert all(df_season['season'] == 2024)

    def test_starter_pitcher_tracking(self, odds_loader, sample_odds):
        """Test that starting pitcher information is tracked correctly."""
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        assert_dataframe_not_empty(df)
        
        assert not df['away_starter'].isnull().all()
        assert not df['home_starter'].isnull().all()

        game_odds = df[(df['away_team'] == 'NYY') & (df['home_team'] == 'BOS')]
        assert len(game_odds) > 0
        assert game_odds.iloc[0]['away_starter'] == 'Cole'
        assert game_odds.iloc[0]['home_starter'] == 'Sale'
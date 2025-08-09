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
from data.loaders.odds_loader import OddsLoader
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
            #  sportsbook, away_odds, home_odds, season)
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'DraftKings', -110, +120, 2024),
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'FanDuel', -115, +125, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'DraftKings', +140, -130, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'FanDuel', +135, -125, 2024),
            ('2024-04-03', '2024-04-03T18:35:00', 'TB', 'BOS', 'Glasnow', 'Pivetta', 'DraftKings', -105, +110, 2024),
        ]
        insert_odds_data(clean_db, odds)
        return odds

    def test_load_for_date_range_basic(self, odds_loader, sample_odds):
        """Test loading odds for a basic date range."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 2)
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, odds_loader.columns)
        assert len(df) == 4  # Should have 4 odds entries in range
        
        # Check date range
        dates = df['game_date'].unique()
        assert '2024-04-01' in dates
        assert '2024-04-02' in dates
        assert '2024-04-03' not in dates

    def test_load_for_date_range_empty(self, odds_loader, sample_odds):
        """Test loading odds for date range with no odds."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 1, 1), 
            end=date(2024, 1, 31)
        )
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_for_date_range_single_day(self, odds_loader, sample_odds):
        """Test loading odds for a single day."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        assert len(df) == 2  # Two sportsbooks for April 1st
        assert all(df['game_date'] == '2024-04-01')

    def test_load_for_season_basic(self, odds_loader, sample_odds):
        """Test loading all odds for a specific season."""
        # Act
        df = odds_loader.load_for_season(season=2024)
        
        # Assert
        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, odds_loader.columns)
        assert len(df) == 5  # All sample odds are from 2024
        
        # Check that all returned odds are from the correct season
        assert all(df['season'] == 2024)
        
        # Check date ordering
        dates = df['game_date'].tolist()
        assert dates == sorted(dates)  # Should be ordered by game_date

    def test_load_for_season_empty(self, odds_loader, sample_odds):
        """Test loading odds for a season with no data."""
        # Act
        df = odds_loader.load_for_season(season=2023)
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_for_season_multiple_seasons(self, odds_loader, clean_db):
        """Test loading odds when multiple seasons exist."""
        # Arrange - Add odds from different seasons
        multi_season_odds = [
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole', 'Sale', 'DraftKings', -110, +120, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'BOS', 'NYY', 'Houck', 'Rodon', 'FanDuel', +140, -130, 2024),
            ('2023-04-01', '2023-04-01T19:05:00', 'TB', 'BOS', 'Glasnow', 'Pivetta', 'DraftKings', -105, +110, 2023),
            ('2023-09-15', '2023-09-15T19:10:00', 'BOS', 'NYY', 'Sale', 'Cole', 'FanDuel', +115, -125, 2023),
            ('2025-04-01', '2025-04-01T19:05:00', 'NYY', 'TB', 'Cole', 'Glasnow', 'DraftKings', -120, +130, 2025),
        ]
        insert_odds_data(clean_db, multi_season_odds)
        
        # Act - Load 2024 season
        df_2024 = odds_loader.load_for_season(season=2024)
        
        # Assert 2024 results
        assert_dataframe_not_empty(df_2024)
        assert len(df_2024) == 2  # Only 2024 odds
        assert all(df_2024['season'] == 2024)
        
        # Act - Load 2023 season  
        df_2023 = odds_loader.load_for_season(season=2023)
        
        # Assert 2023 results
        assert_dataframe_not_empty(df_2023)
        assert len(df_2023) == 2  # Only 2023 odds
        assert all(df_2023['season'] == 2023)

    def test_load_up_to_game_basic(self, odds_loader, sample_odds):
        """Test loading odds for a team up to a specific date."""
        # Act
        df = odds_loader.load_up_to_game(
            date=date(2024, 4, 3), 
            team_abbr='BOS'
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        assert 'team_side' in df.columns
        # Should have odds for games before 4/3 involving BOS
        boston_games = df[(df['home_team'] == 'BOS') | (df['away_team'] == 'BOS')]
        assert len(boston_games) > 0

    def test_load_game_odds_with_sportsbook(self, odds_loader, sample_odds):
        """Test loading odds for a specific game and sportsbook."""
        # Act
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
        
        # Assert
        assert_dataframe_not_empty(df)
        assert len(df) == 1  # Should be exactly one result
        assert df.iloc[0]['sportsbook'] == 'DraftKings'
        assert df.iloc[0]['away_odds'] == -110
        assert df.iloc[0]['home_odds'] == 120

    def test_load_game_odds_all_sportsbooks(self, odds_loader, sample_odds):
        """Test loading odds for a specific game from all sportsbooks."""
        # Act
        df = odds_loader.load_game_odds(
            game_date=date(2024, 4, 1),
            away_team='NYY',
            home_team='BOS',
            away_starter='Cole', 
            home_starter='Sale'
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        assert len(df) == 2  # DraftKings and FanDuel
        sportsbooks = df['sportsbook'].tolist()
        assert 'DraftKings' in sportsbooks
        assert 'FanDuel' in sportsbooks

    def test_load_game_odds_no_results(self, odds_loader, sample_odds):
        """Test loading odds for a game that doesn't exist."""
        # Act
        df = odds_loader.load_game_odds(
            game_date=date(2024, 12, 31),
            away_team='NONE',
            home_team='ALSO_NONE',
            away_starter='Nobody',
            home_starter='No one'
        )
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_column_validation(self, odds_loader, sample_odds):
        """Test that all expected columns are present and valid."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        # Assert
        assert_dataframe_schema(df, odds_loader.columns)
        
        # Check specific column types
        assert pd.api.types.is_datetime64_ns_dtype(df['game_date'])
        assert pd.api.types.is_object_dtype(df['sportsbook'])
        assert pd.api.types.is_numeric_dtype(df['away_odds'])
        assert pd.api.types.is_numeric_dtype(df['home_odds'])
        assert pd.api.types.is_integer_dtype(df['season'])
        
        # Check for no null values in required fields
        required_fields = ['game_date', 'away_team', 'home_team', 'sportsbook']
        for field in required_fields:
            assert not df[field].isnull().any(), f"Field {field} should not have null values"

    def test_odds_values_realistic(self, odds_loader, sample_odds):
        """Test that odds values are within realistic ranges."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        
        # Check that odds are within reasonable ranges
        # Typically odds range from -1000 to +1000
        assert df['away_odds'].min() >= -1000
        assert df['away_odds'].max() <= 1000
        assert df['home_odds'].min() >= -1000
        assert df['home_odds'].max() <= 1000

    @pytest.mark.parametrize("team,expected_min_games", [
        ("BOS", 2),  # Boston appears in at least 2 games
        ("NYY", 2),  # Yankees appear in at least 2 games  
        ("TB", 1),   # Tampa Bay appears in at least 1 game
    ])
    def test_team_odds_availability(self, odds_loader, sample_odds, team, expected_min_games):
        """Test that teams have odds available for expected number of games."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )
        
        # Filter for team games
        team_odds = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
        # Assert
        unique_games = team_odds[['game_date', 'away_team', 'home_team']].drop_duplicates()
        assert len(unique_games) >= expected_min_games

    def test_sportsbook_consistency(self, odds_loader, sample_odds):
        """Test that sportsbook data is consistent."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 2)
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        
        # Check that we have multiple sportsbooks
        sportsbooks = df['sportsbook'].unique()
        assert len(sportsbooks) >= 2
        assert 'DraftKings' in sportsbooks
        assert 'FanDuel' in sportsbooks

    def test_season_filtering(self, odds_loader, clean_db):
        """Test that season filtering works correctly."""
        # Arrange - Add odds from different seasons
        multi_season_odds = [
            ('2024-04-01', '2024-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -110, +110, 2024),
            ('2023-04-01', '2023-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -120, +120, 2023),
            ('2025-04-01', '2025-04-01T19:05:00', 'A', 'B', 'P1', 'P2', 'Book1', -105, +105, 2025),
        ]
        insert_odds_data(clean_db, multi_season_odds)
        
        # Act - Test date range filtering
        df_date_range = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        # Assert date range filtering
        assert_dataframe_not_empty(df_date_range)
        assert len(df_date_range) == 1  # Only 2024 game
        assert df_date_range.iloc[0]['season'] == 2024

        # Act - Test season filtering
        df_season = odds_loader.load_for_season(season=2024)
        
        # Assert season filtering
        assert_dataframe_not_empty(df_season)
        assert len(df_season) == 1  # Only 2024 odds
        assert all(df_season['season'] == 2024)

    def test_starter_pitcher_tracking(self, odds_loader, sample_odds):
        """Test that starting pitcher information is tracked correctly."""
        # Act
        df = odds_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        # Assert
        assert_dataframe_not_empty(df)
        
        # Check that starting pitchers are recorded
        assert not df['away_starter'].isnull().all()
        assert not df['home_starter'].isnull().all()
        
        # Check specific values
        game_odds = df[(df['away_team'] == 'NYY') & (df['home_team'] == 'BOS')]
        assert len(game_odds) > 0
        assert game_odds.iloc[0]['away_starter'] == 'Cole'
        assert game_odds.iloc[0]['home_starter'] == 'Sale'
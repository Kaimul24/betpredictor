"""
Tests for GameLoader class.

Tests the schedule/game data loading functionality including:
- Loading games for date ranges
- Loading team-specific games up to a date
- Season game loading
- Team records and statistics
- Game streaks
- Rest days calculation
"""

import pytest
import pandas as pd
from datetime import date
from data.loaders.game_loader import GameLoader
from tests.conftest import (
    insert_schedule_games, assert_dataframe_schema, 
    assert_dataframe_not_empty, assert_dataframe_values,
    insert_park_factors
)


class TestGameLoader:
    """Test suite for GameLoader class."""
    
    @pytest.fixture
    def game_loader(self, clean_db):
        """Create a GameLoader instance with clean database."""
        return GameLoader()
    
    @pytest.fixture
    def sample_games(self, clean_db):
        """Sample game data for testing."""
        games = [
           #(game_id, game_date, game_datetime, season, away_team, home_team, 
            #    status, away_score, home_score, winning_team, losing_team, dh)
            ('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS', 
             'Final', 2, 8, 'NYY', 'BOS', 0),
            ('game2', '2024-04-02', '2024-04-02T19:10:00', 2024, 'TB', 'NYY', 
             'Final', 1, 5, 'NYY', 'TB', 0),
            ('game3', '2024-04-03', '2024-04-03T18:35:00', 2024, 'TB', 'BOS', 
             'Final', 4, 6, 'BOS', 'TB', 0),
            ('game4', '2024-04-04', '2024-04-04T19:05:00', 2024, 'BOS', 'TB', 
             'Final', 7, 2, 'BOS', 'TB', 0),
            ('game5', '2024-04-05', '2024-04-05T19:10:00', 2024, 'NYY', 'TB', 
             'Final', 3, 1, 'NYY', 'TB', 0),
        ]
        insert_schedule_games(clean_db, games)
        return games

    def test_load_for_date_range_basic(self, game_loader, sample_games):
        """Test loading games for a basic date range."""
        df = game_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 3)
        )

        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, game_loader.columns)
        assert len(df) == 3
        assert_dataframe_values(df, 'game_id', ['game1', 'game2', 'game3'])

    def test_load_for_date_range_empty(self, game_loader, sample_games):
        """Test loading games for date range with no games."""
        df = game_loader.load_for_date_range(
            start=date(2024, 1, 1), 
            end=date(2024, 1, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_for_date_range_single_day(self, game_loader, sample_games):
        """Test loading games for a single day."""
        df = game_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 1)
        )
        
        assert_dataframe_not_empty(df)
        assert len(df) == 1
        assert df.iloc[0]['game_id'] == 'game1'

    def test_load_up_to_game_basic(self, game_loader, sample_games):
        """Test loading games for a team up to a specific date."""
        df = game_loader.load_up_to_game(
            date=date(2024, 4, 3), 
            team_abbr='BOS'
        )
        
        assert_dataframe_not_empty(df)
        assert 'team_side' in df.columns
        assert len(df) == 1

    def test_load_season_games_with_team(self, game_loader, sample_games):
        """Test loading all games for a team in a season."""
        df = game_loader.load_season_games(season=2024, team_abbr='NYY')
        
        assert_dataframe_not_empty(df)
        assert len(df) == 3 

        nyy_games = df[(df['home_team'] == 'NYY') | (df['away_team'] == 'NYY')]
        assert len(nyy_games) == len(df)

    def test_load_season_games_all_teams(self, game_loader, sample_games):
        """Test loading all games for a season without team filter."""
        df = game_loader.load_season_games(season=2024)
        
        assert_dataframe_not_empty(df)
        assert len(df) == 5

    def test_load_up_to_game_season(self, game_loader, sample_games):
        """Test loading team games up to date within season only."""
        df = game_loader.load_up_to_game_season(
            date=date(2024, 4, 4), 
            team_abbr='BOS'
        )
        
        assert_dataframe_not_empty(df)
        assert 'team_side' in df.columns
        assert 'team_won' in df.columns
        assert len(df) >= 2

    def test_team_participation(self, game_loader, sample_games):
        """Test that teams appear in expected number of games."""
        df = game_loader.load_season_games(season=2024, team_abbr='NYY')
        assert len(df) >= 2
        
        df = game_loader.load_season_games(season=2024, team_abbr='BOS')
        assert len(df) >= 3

    def test_load_for_date_range_column_validation(self, game_loader, sample_games):
        """Test that all expected columns are present and valid."""
        df = game_loader.load_for_date_range(
            start=date(2024, 4, 1), 
            end=date(2024, 4, 5)
        )
        
        assert_dataframe_schema(df, game_loader.columns)
        
        assert pd.api.types.is_object_dtype(df['game_id'])
        assert pd.api.types.is_datetime64_ns_dtype(df['game_date'])
        assert pd.api.types.is_integer_dtype(df['season'])
        
        required_fields = ['game_id', 'game_date', 'away_team', 'home_team']
        for field in required_fields:
            assert not df[field].isnull().any(), f"Field {field} should not have null values"

    def test_doubleheader_handling(self, game_loader, clean_db):
        """Test handling of doubleheader games."""
        #(game_id, game_date, game_datetime, season, away_team, home_team, 
        #        status, away_score, home_score, winning_team, losing_team, dh)
        dh_games = [
            ('dh1', '2024-04-10', '2024-04-10T13:05:00', 2024, 'Team A', 'Team B', 'Final', 3, 2, 'A', 'B', 1),
            ('dh2', '2024-04-10', '2024-04-10T19:05:00', 2024, 'Team A', 'Team B', 'Final', 1, 5, 'B', 'A', 2),
        ]
        insert_schedule_games(clean_db, dh_games)
        
        df = game_loader.load_for_date_range(
            start=date(2024, 4, 10), 
            end=date(2024, 4, 10)
        )
        
        assert len(df) == 2
        assert df['dh'].tolist() == [1, 2]

    def test_load_park_factor_season_basic(self, game_loader, clean_db):
        """Test loading park factors for a season."""
        park_factors = [
            (1, 'Fenway Park', 2024, 105),
            (2, 'Yankee Stadium', 2024, 98),
            (3, 'Coors Field', 2024, 115),
            (4, 'Petco Park', 2024, 92),
        ]
        insert_park_factors(clean_db, park_factors)
        
        df = game_loader.load_park_factor_season(season=2024)
        
        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, game_loader.venue_columns)
        assert len(df) == 4
        assert_dataframe_values(df, 'venue_name', ['Fenway Park', 'Yankee Stadium', 'Coors Field', 'Petco Park'])
        assert_dataframe_values(df, 'park_factor', [105, 98, 115, 92])

    def test_load_park_factor_season_empty(self, game_loader, clean_db):
        """Test loading park factors for season with no data."""
        df = game_loader.load_park_factor_season(season=2023)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_park_factor_season_filters_null(self, game_loader, clean_db):
        """Test that park factors with NULL values are filtered out."""
        park_factors = [
            (1, 'Fenway Park', 2024, 105),
            (2, 'Yankee Stadium', 2024, None),  # This should be filtered out
            (3, 'Coors Field', 2024, 115),
        ]
        
        # Insert manually to include NULL value
        query = """
        INSERT INTO park_factors (venue_id, venue_name, season, park_factor) 
        VALUES (?, ?, ?, ?)
        """
        clean_db.execute_many_write_queries(query, park_factors)
        
        df = game_loader.load_park_factor_season(season=2024)
        
        assert_dataframe_not_empty(df)
        assert len(df) == 2  # Only non-NULL park factors
        assert_dataframe_values(df, 'venue_name', ['Fenway Park', 'Coors Field'])

    def test_load_park_factor_season_multiple_seasons(self, game_loader, clean_db):
        """Test loading park factors with multiple seasons."""
        park_factors = [
            (1, 'Fenway Park', 2023, 102),
            (1, 'Fenway Park', 2024, 105),
            (2, 'Yankee Stadium', 2023, 95),
            (2, 'Yankee Stadium', 2024, 98),
        ]
        insert_park_factors(clean_db, park_factors)
        
        # Test 2024 season
        df_2024 = game_loader.load_park_factor_season(season=2024)
        assert len(df_2024) == 2
        assert_dataframe_values(df_2024, 'season', [2024, 2024])
        assert_dataframe_values(df_2024, 'park_factor', [105, 98])
        
        # Test 2023 season
        df_2023 = game_loader.load_park_factor_season(season=2023)
        assert len(df_2023) == 2
        assert_dataframe_values(df_2023, 'season', [2023, 2023])
        assert_dataframe_values(df_2023, 'park_factor', [102, 95])

    def test_load_park_factor_season_column_validation(self, game_loader, clean_db):
        """Test that all expected columns are present and valid."""
        park_factors = [
            (1, 'Fenway Park', 2024, 105),
            (2, 'Yankee Stadium', 2024, 98),
        ]
        insert_park_factors(clean_db, park_factors)
        
        df = game_loader.load_park_factor_season(season=2024)
        
        assert_dataframe_schema(df, game_loader.venue_columns)
        
        # Check data types
        assert pd.api.types.is_integer_dtype(df['venue_id'])
        assert pd.api.types.is_object_dtype(df['venue_name'])
        assert pd.api.types.is_integer_dtype(df['season'])
        assert pd.api.types.is_integer_dtype(df['park_factor'])
        
        # Check required fields have no null values
        required_fields = ['venue_id', 'venue_name', 'season', 'park_factor']
        for field in required_fields:
            assert not df[field].isnull().any(), f"Field {field} should not have null values"
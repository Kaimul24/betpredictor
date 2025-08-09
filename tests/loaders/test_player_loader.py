"""
Tests for PlayerLoader class.

Tests for player statistics loading functionality including:
- Loading batting and pitching stats
- Player performance over time windows
- Lineup loading with stats
- Aggregation functions

Note: PlayerLoader is currently empty/placeholder. These tests are prepared
for when the loader is implemented.
"""

import pytest
import pandas as pd
from datetime import date
from data.loaders.player_loader import PlayerLoader
from tests.conftest import (
    insert_players, insert_batting_stats, insert_pitching_stats,
    insert_lineup_players, assert_dataframe_schema, 
    assert_dataframe_not_empty, assert_dataframe_values
)


class TestPlayerLoader:
    """Test suite for PlayerLoader class."""
    
    @pytest.fixture
    def player_loader(self, clean_db):
        """Create a PlayerLoader instance with clean database."""
        return PlayerLoader()
    
    @pytest.fixture
    def sample_players(self, clean_db):
        """Insert sample player data for testing."""
        players = [
            # (player_id, name, pos, current_team)
            ('player1', 'Mike Trout', 'OF', 'LAA'),
            ('player2', 'Shohei Ohtani', 'DH', 'LAA'), 
            ('player3', 'Mookie Betts', 'OF', 'LAD'),
            ('player4', 'Rafael Devers', '3B', 'SFG'),
        ]
        insert_players(clean_db, players)
        return players
    
    @pytest.fixture
    def sample_batting_stats(self, clean_db, sample_players):
        """Insert sample batting stats for testing."""
        stats = [
            # (player_id, game_date, team, dh, ab, pa, ops, wrc_plus, season)
            ('player1', '2024-04-01', 'LAA', 0, 4, 5, 0.950, 150, 2024),
            ('player1', '2024-04-02', 'LAA', 0, 3, 4, 0.880, 140, 2024),
            ('player2', '2024-04-01', 'LAA', 0, 4, 4, 1.100, 180, 2024),
            ('player3', '2024-04-01', 'LAD', 0, 5, 5, 0.920, 135, 2024),
            ('player4', '2024-04-02', 'SFG', 1, 4, 5, 0.820, 115, 2024),
            ('player4', '2024-04-02', 'SFG', 2, 4, 5, 0.820, 115, 2024) 
        ]
        insert_batting_stats(clean_db, stats)
        return stats
    
    @pytest.fixture
    def sample_pitching_stats(self, clean_db, sample_players):
        """Insert sample pitching stats for testing."""
        stats = [
            # (player_id, game_date, team, dh, era, ip, k_percent, season)
            ('player4', '2024-04-01', 'TEX', 0, 2.50, 6.0, 32.5, 2024),
            ('player4', '2024-04-05', 'TEX', 0, 2.20, 7.0, 35.0, 2024),
        ]
        insert_pitching_stats(clean_db, stats)
        return stats
    
    def test_load_for_season_basic(self, player_loader, sample_batting_stats):
        """Test loading batting stats for a season"""
        df = player_loader.load_for_season(season=2024)
        assert_dataframe_not_empty(df)
        assert len(df) == 6

    def test_load_for_season_invalid_season(self, player_loader, sample_batting_stats):
        """Test loading batting stats for an invalid season"""
        df = player_loader.load_for_season(season=2020)
        assert len(df) == 0

    def test_load_batter_stats_basic(self, player_loader, sample_batting_stats):
        """Test loading basic batting stats for a player."""

        df = player_loader.load_batter_stats(player_id='player1', season=2024)
        assert_dataframe_not_empty(df)
        assert 'wrc_plus' in df.columns
        assert len(df) == 2 

    def test_load_pitcher_stats_basic(self, player_loader, sample_pitching_stats):
        """Test loading basic pitching stats for a player."""

        df = player_loader.load_pitcher_stats(player_id='player4', season=2024)
        assert_dataframe_not_empty(df)
        assert 'era' in df.columns
        assert len(df) == 2

    def test_load_batting_stats_for_date_range_basic(self, player_loader, sample_batting_stats):
        """Test loading player stats for a date range."""
        df = player_loader.load_batting_stats_for_date_range(start=date(2024, 4, 1), end=date(2024, 4, 2))

        assert_dataframe_not_empty(df)
        assert len(df) == 6

    def test_load_batting_stats_for_date_range_team(self, player_loader, sample_batting_stats):
        """Test loading player stats for a date range for a given team."""
        df = player_loader.load_batting_stats_for_date_range(start=date(2024, 4, 1), end=date(2024, 4, 2), team_abbr='LAA')

        assert_dataframe_not_empty(df)
        assert len(df) == 3

    def test_load_batting_stats_up_to_game(self, player_loader, sample_batting_stats):
        before_first_game = player_loader.load_batting_stats_up_to_game(date=date(2024, 4, 2),
                            team_abbr='SFG', dh=1)
        
        assert len(before_first_game) == 0

        before_second_game = player_loader.load_batting_stats_up_to_game(date=date(2024, 4, 2),
                            team_abbr='SFG', dh=2)
        
        assert_dataframe_not_empty(before_second_game)
        assert len(before_second_game) == 1

        after_second_game = player_loader.load_batting_stats_up_to_game(date=date(2024, 4, 3),
                            team_abbr='SFG', dh=0)
        
        assert_dataframe_not_empty(before_second_game)
        assert len(after_second_game) == 2

    @pytest.mark.skip(reason="PlayerLoader not yet implemented")
    def test_player_vs_pitcher_matchup(self, player_loader, sample_batting_stats, sample_pitching_stats):
        """Test loading historical matchup data between batter and pitcher."""
        pass

    @pytest.mark.skip(reason="PlayerLoader not yet implemented")
    def test_load_player_splits(self, player_loader, sample_batting_stats):
        """Test loading player performance splits (home/away, vs handedness, etc)."""
        pass
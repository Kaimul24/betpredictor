"""
Tests for TeamLoader class.

Tests for team-level statistics and analysis functionality including:
- Team records and standings
- Bullpen usage patterns  
- Strength of schedule calculations
- Team trends and momentum
- Aggregated player stats to team level

Note: TeamLoader is currently empty/placeholder. These tests are prepared
for when the loader is implemented.
"""

import pytest
import pandas as pd
from datetime import date
from data.loaders.team_loader import TeamLoader
from tests.conftest import (
    insert_schedule_games, insert_batting_stats, insert_rosters, insert_lineup_players,
    assert_dataframe_schema, assert_dataframe_not_empty, assert_dataframe_values
)


class TestTeamLoader:
    """Test suite for TeamLoader class."""
    
    @pytest.fixture
    def team_loader(self, clean_db):
        """Create a TeamLoader instance with clean database."""
        return TeamLoader()
    
    @pytest.fixture
    def sample_team_games(self, clean_db):
        """Insert sample team game data for testing."""
        games = [
            # (game_id, game_date, season, away_team, home_team, away_abbr, home_abbr, 
            #  status, away_score, home_score, winning_team, losing_team)
            ('game1', '2024-04-01', 2024, 'Yankees', 'Red Sox', 'NYY', 'BOS', 'Final', 5, 3, 'NYY', 'BOS'),
            ('game2', '2024-04-02', 2024, 'Red Sox', 'Yankees', 'BOS', 'NYY', 'Final', 2, 8, 'NYY', 'BOS'),
            ('game3', '2024-04-03', 2024, 'Rays', 'Red Sox', 'TB', 'BOS', 'Final', 4, 6, 'BOS', 'TB'),
            ('game4', '2024-04-04', 2024, 'Red Sox', 'Rays', 'BOS', 'TB', 'Final', 7, 2, 'BOS', 'TB'),
            ('game5', '2024-04-05', 2024, 'Yankees', 'Rays', 'NYY', 'TB', 'Final', 3, 1, 'NYY', 'TB'),
        ]
        insert_schedule_games(clean_db, games)
        return games
    
    @pytest.fixture
    def sample_team_batting_stats(self, clean_db):
        """Insert sample team batting stats for testing."""
        stats = [
            # (player_id, game_date, team, dh, ab, pa, ops, wrc_plus, season)
            ('player1', '2024-04-01', 'NYY', 0, 4, 5, 0.950, 150, 2024),
            ('player2', '2024-04-01', 'NYY', 0, 3, 4, 0.880, 140, 2024),
            ('player3', '2024-04-01', 'BOS', 0, 4, 4, 1.100, 180, 2024),
            ('player4', '2024-04-01', 'BOS', 0, 5, 5, 0.920, 135, 2024),
        ]
        insert_batting_stats(clean_db, stats)
        return stats
    
    @pytest.fixture
    def sample_lineup(self, clean_db):
        lineup = [
            # (game_date, team_id, team, dh, player_id, position, batting_order, season)
            ('2024-04-01', 30, 'SFG', 0, 'player1', 'DH', 1, '2024'),
            ('2024-04-01', 30, 'SFG', 0, 'player2', 'C',  2, '2024'),
            ('2024-04-01', 30, 'SFG', 0, 'player3', '1B', 3, '2024'),
            ('2024-04-01', 30, 'SFG', 0, 'player4', '2B', 4, '2024'),
            ('2024-04-01', 12,'LAA', 0, 'player5', '3B',  1,'2024'),
            ('2024-04-01', 12,'LAA', 0, 'player6', 'SS',  2, '2024'),
            ('2024-04-01', 12,'LAA', 0, 'player7', 'LF',  3,'2024'),
            ('2024-04-01', 12,'LAA', 0, 'player8', 'CF',  4,'2024'),
            ('2024-04-02', 30, 'SFG', 0, 'player1', 'DH', 1,'2024'),
            ('2024-04-02', 30, 'SFG', 0, 'player2', 'C',  2,'2024'),
            ('2024-04-02', 30, 'SFG', 0, 'player3', '1B', 3,'2024'),
            ('2024-04-02', 30, 'SFG', 0, 'player4', '2B', 4,'2024'),
            ('2024-04-02', 12, 'LAA', 0, 'player5', '3B', 1,'2024'),
            ('2024-04-02', 12, 'LAA', 0, 'player6', 'SS', 2, '2024'),
            ('2024-04-02', 12, 'LAA', 0, 'player7', 'LF', 3,'2024'),
            ('2024-04-02', 12, 'LAA', 0, 'player8', 'CF', 4,'2024'),
        ]
        insert_lineup_players(clean_db, lineup)
        return lineup
    
    @pytest.fixture
    def sample_roster(self, clean_db):
        roster = [
            # (game_date, season, team, player_name, position, status)
            ('2024-04-01', 2024, 'SFG', 'Jung Hoo Lee', 'CF', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Heliot Ramos', 'LF', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Rafael Devers', 'DH', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Willy Adames', 'SS', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Matt Chapman', '3B', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Zack Neto', 'SS', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Nolan Schanuel', '1B', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Mike Trout', 'DH', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Taylor Ward', 'LF', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Jo Adell', 'CF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Jung Hoo Lee', 'CF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Heliot Ramos', 'LF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Rafael Devers', 'DH', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Willy Adames', 'SS', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Matt Chapman', '3B', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Zack Neto', 'SS', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Nolan Schanuel', '1B', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Mike Trout', 'DH', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Taylor Ward', 'LF', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Jo Adell', 'CF', 'Active')
        ]
        insert_rosters(clean_db, roster)
        return roster
    
    def test_load_lineup_basic(self, team_loader, sample_lineup):
        """Test loading lineup information for a season"""
        df = team_loader.load_lineup(season = 2024)
        assert len(df) == 16

    def test_load_lineup_team_basic(self, team_loader, sample_lineup):
        """Test loading lineup information for a team"""
        df = team_loader.load_lineup(
            team = 'SFG', season = 2024
        )

        assert_dataframe_not_empty(df)
        assert 'batting_order' in df.columns
        assert len(df) == 8

    def test_load_lineup_invalid_team_season(self, team_loader, sample_lineup):
        """Test invalid team lineup"""
        df = team_loader.load_lineup(team = 'SFG', season = 2020)
        assert len(df) == 0

        df = team_loader.load_lineup(team = 'NAN', season = 2024)
        assert len(df) == 0
        
    def test_load_lineup_date(self, team_loader, sample_lineup):
        """Test loading lineup information with date"""
        df = team_loader.load_lineup(
            team = 'SFG', season = 2024, date=date(2024, 4, 1)
        )

        assert_dataframe_not_empty(df)
        assert 'batting_order' in df.columns
        assert len(df) == 4

    def test_load_roster_basic(self, team_loader, sample_roster):
        "Test loading roster information in a season"
        df = team_loader.load_roster(season=2024)

        assert_dataframe_not_empty(df)
        assert_dataframe_schema(df, ['game_date', 'team', 'player_name', 'position', 'status'])
        assert len(df) == 20

    def test_load_roster_team(self, team_loader, sample_roster):
        "Test loading roster information in a season for a specific team"
        df = team_loader.load_roster(season=2024, team='SFG')

        assert_dataframe_not_empty(df)
        assert len(df) == 10

    def test_load_roster_date(self, team_loader, sample_roster):
        """Test loading roster information for all teams on a specific date"""
        df = team_loader.load_roster(season=2024, date=date(2024, 4, 1))

        assert_dataframe_not_empty(df)
        assert len(df) == 10

    def test_load_roster_invalid_season(self, team_loader, sample_roster):
        """Test loading roster to ensure ValueError is raised when season and year of date do not match"""
        with pytest.raises(ValueError):
            df = team_loader.load_roster(season=2024, date=date(2023, 4, 1))

    def test_load_roster_no_data(self, team_loader, sample_roster):
        """Test loading roster when there is no season data"""
        df = team_loader.load_roster(season=2021)
        
        assert len(df) == 0

    # def test_team_record_basic(self, team_loader, sample_team_games):
    #     """Test calculating team record up to a date."""
    #     record = team_loader.team_record(date=date(2024, 4, 5), team_abbr='BOS')
        
    #     assert isinstance(record, dict)
    #     assert 'wins' in record
    #     assert 'losses' in record
    #     assert 'games' in record
    #     assert 'win_pct' in record
        
    #     assert isinstance(record['wins'], int)
    #     assert isinstance(record['losses'], int)
    #     assert isinstance(record['games'], int)
    #     assert isinstance(record['win_pct'], float)
        
    #     assert record['wins'] + record['losses'] == record['games']
    #     if record['games'] > 0:
    #         expected_pct = record['wins'] / record['games']
    #         assert abs(record['win_pct'] - expected_pct) < 0.001

    # def test_team_record_no_games(self, team_loader, clean_db):
    #     """Test team record when team has no games."""
    #     record = team_loader.team_record(date=date(2024, 4, 5), team_abbr='NONE')
        
    #     assert record == {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'games': 0}

    # def test_game_streak_basic(self, team_loader, sample_team_games):
    #     """Test calculating game streak for a team."""
    #     streak = team_loader.game_streak(date=date(2024, 4, 6), team_abbr='NYY')
        
    #     assert isinstance(streak, int)
    #     assert streak == 3

    # def test_game_streak_no_games(self, team_loader, clean_db):
    #     """Test game streak when team has no games."""
    #     streak = team_loader.game_streak(date=date(2024, 4, 5), team_abbr='NONE')
        
    #     assert streak == 0

    # def test_rest_days_basic(self, team_loader, sample_team_games):
    #     """Test calculating rest days for a team."""
    #     rest = team_loader.rest_days(date=date(2024, 4, 10), team_abbr='NYY')
        
    #     assert isinstance(rest, int)
    #     assert rest >= 0

    # def test_rest_days_no_previous_games(self, team_loader, clean_db):
    #     """Test rest days when team has no previous games."""
    #     rest = team_loader.rest_days(date=date(2024, 4, 1), team_abbr='NONE')
        
    #     assert rest == 0

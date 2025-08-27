import pytest
from datetime import date
from data.loaders.team_loader import TeamLoader
from tests.conftest import (
    insert_schedule_games, insert_batting_stats, insert_rosters, insert_lineup_players,
    insert_lineups, assert_dataframe_schema, assert_dataframe_not_empty, assert_dataframe_values
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
        # First insert lineup data (team-level info)
        lineup_data = [
            # (game_date, team_id, team, dh, opposing_team_id, opposing_team, team_starter_id, season)
            ('2024-04-01', 30, 'SFG', 0, 12, 'LAA', 'starter1', 2024),
            ('2024-04-01', 12, 'LAA', 0, 30, 'SFG', 'starter2', 2024),
            ('2024-04-02', 30, 'SFG', 0, 12, 'LAA', 'starter3', 2024),
            ('2024-04-02', 12, 'LAA', 0, 30, 'SFG', 'starter4', 2024),
        ]
        insert_lineups(clean_db, lineup_data)
        
        # Then insert lineup player data
        lineup = [
            # (game_date, team_id, team, opposing_team_id, opposing_team, dh, player_id, position, batting_order, season)
            ('2024-04-01', 30, 'SFG', 12, 'LAA', 0, 'player1', 'DH', 1, '2024'),
            ('2024-04-01', 30, 'SFG', 12, 'LAA', 0, 'player2', 'C',  2, '2024'),
            ('2024-04-01', 30, 'SFG', 12, 'LAA', 0, 'player3', '1B', 3, '2024'),
            ('2024-04-01', 30, 'SFG', 12, 'LAA', 0, 'player4', '2B', 4, '2024'),
            ('2024-04-01', 12,'LAA', 30, 'SFG', 0, 'player5', '3B',  1,'2024'),
            ('2024-04-01', 12,'LAA', 30, 'SFG', 0, 'player6', 'SS',  2, '2024'),
            ('2024-04-01', 12,'LAA', 30, 'SFG', 0, 'player7', 'LF',  3,'2024'),
            ('2024-04-01', 12,'LAA', 30, 'SFG', 0, 'player8', 'CF',  4,'2024'),
            ('2024-04-02', 30, 'SFG', 12, 'LAA', 0, 'player1', 'DH', 1,'2024'),
            ('2024-04-02', 30, 'SFG', 12, 'LAA', 0, 'player2', 'C',  2,'2024'),
            ('2024-04-02', 30, 'SFG', 12, 'LAA', 0, 'player3', '1B', 3,'2024'),
            ('2024-04-02', 30, 'SFG', 12, 'LAA', 0, 'player4', '2B', 4,'2024'),
            ('2024-04-02', 12, 'LAA', 30, 'SFG', 0, 'player5', '3B', 1,'2024'),
            ('2024-04-02', 12, 'LAA', 30, 'SFG', 0, 'player6', 'SS', 2, '2024'),
            ('2024-04-02', 12, 'LAA', 30, 'SFG', 0, 'player7', 'LF', 3,'2024'),
            ('2024-04-02', 12, 'LAA', 30, 'SFG', 0, 'player8', 'CF', 4,'2024'),
        ]
        insert_lineup_players(clean_db, lineup)
        return lineup_data
    
    @pytest.fixture
    def sample_roster(self, clean_db):
        roster = [
            # (game_date, season, team, player_name, player_id, position, status)
            ('2024-04-01', 2024, 'SFG', 'Jung Hoo Lee', 1001, 'CF', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Heliot Ramos', 1002, 'LF', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Rafael Devers', 1003, 'DH', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Willy Adames', 1004, 'SS', 'Active'),
            ('2024-04-01', 2024, 'SFG', 'Matt Chapman', 1005, '3B', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Zack Neto', 2001, 'SS', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Nolan Schanuel', 2002, '1B', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Mike Trout', 2003, 'DH', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Taylor Ward', 2004, 'LF', 'Active'),
            ('2024-04-01', 2024, 'LAA', 'Jo Adell', 2005, 'CF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Jung Hoo Lee', 1001, 'CF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Heliot Ramos', 1002, 'LF', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Rafael Devers', 1003, 'DH', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Willy Adames', 1004, 'SS', 'Active'),
            ('2024-04-02', 2024, 'SFG', 'Matt Chapman', 1005, '3B', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Zack Neto', 2001, 'SS', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Nolan Schanuel', 2002, '1B', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Mike Trout', 2003, 'DH', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Taylor Ward', 2004, 'LF', 'Active'),
            ('2024-04-02', 2024, 'LAA', 'Jo Adell', 2005, 'CF', 'Active')
        ]
        insert_rosters(clean_db, roster)
        return roster
    
    def test_load_pitching_matchups_basic(self, team_loader, sample_lineup):
        """Test loading pitching matchups for a season"""
        df = team_loader.load_pitching_matchups(season = 2024)
        assert len(df) == 4

    def test_load_pitching_matchups_team_basic(self, team_loader, sample_lineup):
        """Test loading pitching matchups for a team"""
        df = team_loader.load_pitching_matchups(season = 2024)

        assert_dataframe_not_empty(df)
        assert 'team_starter_id' in df.columns
        assert len(df) == 4

    def test_load_pitching_matchups_invalid_season(self, team_loader, sample_lineup):
        """Test invalid season for pitching matchups"""
        df = team_loader.load_pitching_matchups(season = 2020)
        assert len(df) == 0
        
    def test_load_pitching_matchups_data_validation(self, team_loader, sample_lineup):
        """Test pitching matchups data validation"""
        df = team_loader.load_pitching_matchups(season = 2024)

        assert_dataframe_not_empty(df)
        assert 'team_starter_id' in df.columns
        assert 'opposing_starter_id' in df.columns
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

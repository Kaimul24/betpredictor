import pytest
from datetime import date
from data.loaders.player_loader import PlayerLoader
from tests.conftest import (
    insert_players, insert_batting_stats, insert_pitching_stats,
    insert_fielding_stats, assert_dataframe_not_empty
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
    
    @pytest.fixture
    def sample_fielding_stats(self, clean_db):
        """Insert sample fielding stats for testing"""
        stats = [
            # (name, season, month, frv, total_innings)
            ('Mike Trout', 2024, 'April', 5.2, 120.0),
            ('Mike Trout', 2024, 'May', 3.8, 130.0),
            ('Mookie Betts', 2024, 'April', 8.1, 125.0),
            ('Mookie Betts', 2024, 'May', 6.5, 135.0),
            ('Rafael Devers', 2024, 'April', -2.1, 110.0),
            ('Jung Hoo Lee', 2023, 'April', 1.2, 15.0),  # Different season
        ]
        insert_fielding_stats(clean_db, stats)
        return stats
    
    def test_load_for_season_basic(self, player_loader, sample_batting_stats):
        """Test loading batting stats for a season"""
        df = player_loader.load_for_season_batter(season=2024)
        assert_dataframe_not_empty(df)
        assert len(df) == 6

    def test_load_for_season_invalid_season(self, player_loader, sample_batting_stats):
        """Test loading batting stats for an invalid season"""
        df = player_loader.load_for_season_batter(season=2020)
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

    def test_load_fielding_stats_basic(self, player_loader, sample_fielding_stats):
        """Test loading fielding stats for a season."""
        df = player_loader.load_fielding_stats(season=2024)
        assert_dataframe_not_empty(df)
        assert len(df) == 5

    def test_load_fielding_stats_invalid_season(self, player_loader, sample_fielding_stats):
        """Test loading fielding stats for an invalid season."""
        df = player_loader.load_fielding_stats(season=2020)
        assert len(df) == 0

    def test_load_fielding_stats_columns(self, player_loader, sample_fielding_stats):
        """Test that fielding stats dataframe has expected columns."""
        df = player_loader.load_fielding_stats(season=2024)
        expected_columns = ['name', 'normalized_player_name', 'season', 'month', 
                           'frv', 'total_innings']
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_fielding_stats_ordering(self, player_loader, sample_fielding_stats):
        """Test that fielding stats are ordered by season, month."""
        df = player_loader.load_fielding_stats(season=2024)
       
        mike_trout_data = df[df['name'] == 'Mike Trout'].reset_index(drop=True)
        assert len(mike_trout_data) == 2
        assert mike_trout_data.iloc[0]['month'] == 'April'
        assert mike_trout_data.iloc[1]['month'] == 'May'

    @pytest.mark.skip(reason="player_vs_pitcher_matchup not yet implemented")
    def test_player_vs_pitcher_matchup(self, player_loader, sample_batting_stats, sample_pitching_stats):
        """Test loading historical matchup data between batter and pitcher."""
        pass

    @pytest.mark.skip(reason="player_splits not yet implemented")
    def test_load_player_splits(self, player_loader, sample_batting_stats):
        """Test loading player performance splits (home/away, vs handedness, etc)."""
        pass
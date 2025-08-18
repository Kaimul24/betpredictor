"""
Tests for BattingFeatures class.

Tests the batting stats rolling features calculation including:
- Rolling window calculations with proper temporal ordering
- Data leakage prevention by ensuring only historical data is used
- Weighted averages for plate appearance dependent metrics
- Simple averages for rate metrics
- Multiple rolling windows (3, 7, 14, 25 games)
- Player batch processing
- Caching functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from data.features.player_features.batting import BattingFeatures
from tests.conftest import (
    insert_batting_stats, insert_players, assert_dataframe_schema,
    assert_dataframe_not_empty
)


class TestBattingFeatures:
    """Test suite for BattingFeatures class."""

    @pytest.fixture
    def batting_features(self, clean_db):
        """Create a BattingFeatures instance with clean database."""
        empty_df = pd.DataFrame()
        return BattingFeatures(season=2024, data=empty_df)

    def _load_batting_data(self, season=2024):
        """Helper method to load batting data from database for testing."""
        from data.database import get_database_manager
        db = get_database_manager()
        
        query = """
        SELECT player_id, game_date, team, dh, ab, pa, ops, wrc_plus, season
        FROM batting_stats 
        WHERE season = ?
        ORDER BY game_date, dh
        """
        
        with db.get_reader_connection() as conn:
            data = pd.read_sql_query(query, conn, params=[season])
        
        return data

    @pytest.fixture 
    def sample_players(self, clean_db):
        """Sample player data for testing."""
        players = [
            ('player1', 'John Doe', 'OF', 'NYY'),
            ('player2', 'Jane Smith', 'IF', 'BOS'), 
            ('player3', 'Bob Johnson', 'C', 'TB')
        ]
        insert_players(clean_db, players)
        return players

    @pytest.fixture
    def temporal_batting_data(self, clean_db, sample_players):
        """
        Sample batting data with temporal ordering for data leakage testing.
        Creates a sequence of games over multiple days with varying stats.
        """
        base_date = date(2024, 4, 1)
        
        stats = []
        
        for i in range(10):
            game_date = base_date + timedelta(days=i)
            ops_value = 1.000 - (i * 0.05)  # Declining from 1.000 to 0.550
            wrc_plus_value = 150 - (i * 5)  # Declining from 150 to 105
            pa_value = 4 + (i % 2)  # Alternating between 4 and 5 PA
            
            stats.append((
                'player1', 
                game_date.strftime('%Y-%m-%d'),
                'NYY',
                0,  # dh
                pa_value,  # ab (simplified, usually less than PA)
                pa_value,  # pa
                ops_value,  # ops
                wrc_plus_value,  # wrc_plus
                2024
            ))
        
        # Player 2: 8 games with improving performance
        for i in range(8):
            game_date = base_date + timedelta(days=i)
            ops_value = 0.600 + (i * 0.04)  # Improving from 0.600 to 0.880
            wrc_plus_value = 80 + (i * 8)  # Improving from 80 to 136
            pa_value = 3 + (i % 3)  # PA between 3-5
            
            stats.append((
                'player2',
                game_date.strftime('%Y-%m-%d'),
                'BOS', 
                0,  # dh
                pa_value,
                pa_value,
                ops_value,
                wrc_plus_value,
                2024
            ))
        
        for i in range(5):
            game_date = base_date + timedelta(days=i*2)
            stats.append((
                'player3',
                game_date.strftime('%Y-%m-%d'),
                'TB',
                0,  # dh  
                4,  # ab
                4,  # pa
                0.750,  # ops
                110,  # wrc_plus
                2024
            ))

        insert_batting_stats(clean_db, stats)
        return stats

    @pytest.fixture
    def doubleheader_batting_data(self, clean_db, sample_players):
        """
        Sample batting data including doubleheader games for testing dh handling.
        """
        base_date = date(2024, 4, 1)
        stats = []
        
        # Player 1: Regular game + doubleheader
        for dh in [0, 1]:  # Game 1 and Game 2 of doubleheader
            stats.append((
                'player1',
                base_date.strftime('%Y-%m-%d'),
                'NYY',
                dh,
                4,  # ab
                4,  # pa
                0.800 + (dh * 0.1),  # Different performance in each game
                120 + (dh * 10),  # wrc_plus
                2024
            ))
        
        next_date = base_date + timedelta(days=1)
        stats.append((
            'player1',
            next_date.strftime('%Y-%m-%d'),
            'NYY',
            0,
            4,
            4,
            0.750,
            110,
            2024
        ))
        
        insert_batting_stats(clean_db, stats)
        return stats

    def test_init(self):
        """Test BattingFeatures initialization."""
        data = pd.DataFrame()
        bf = BattingFeatures(season=2024, data=data)
        
        assert bf.season == 2024
        assert bf.rolling_windows == [25, 14, 9, 5, 3]
        assert len(bf.rolling_metrics) == 13
        assert 'ops' in bf.rolling_metrics
        assert 'wrc_plus' in bf.rolling_metrics

    def test_load_data_empty_season(self, batting_features):
        """Test loading batting data with no data for season."""
        data = self._load_batting_data(2024)
        assert data.empty, "Should return empty DataFrame when no batting data exists"

    def test_load_data_with_data(self, batting_features, temporal_batting_data):
        """Test loading batting data returns expected data structure."""
        data = self._load_batting_data(2024)
        
        assert_dataframe_not_empty(data)
        expected_columns = ['player_id', 'game_date', 'team', 'dh', 'ab', 'pa', 'ops', 'wrc_plus', 'season']
        assert_dataframe_schema(data, expected_columns)
        
        assert len(data) == len(temporal_batting_data)
        assert data['season'].iloc[0] == 2024

    def test_rolling_window_calculation_temporal_validation(self, batting_features, temporal_batting_data):
        """
        Test that rolling calculations only use historical data (no data leakage).
        Verify that current game stats are not included in rolling calculations.
        """
        data = self._load_batting_data(2024)
        player1_data = data[data['player_id'] == 'player1'].copy()
        player1_data['game_date'] = pd.to_datetime(player1_data['game_date'])
        player1_data = player1_data.sort_values(['game_date', 'dh'])
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player1_data)
        
        window_3_stats = rolling_stats[rolling_stats['window_size'] == 3].copy()
        window_3_stats = window_3_stats.sort_values(['game_date', 'dh'])
        
        first_game = window_3_stats.iloc[0]
        assert pd.isna(first_game['ops_rolling']), "First game should have no rolling stats"
        
        second_game = window_3_stats.iloc[1]
        first_game_ops = player1_data.iloc[0]['ops']
        assert abs(second_game['ops_rolling'] - first_game_ops) < 0.001, \
            "Second game rolling should equal first game's stats"
        
        if len(window_3_stats) >= 4:
            fourth_game = window_3_stats.iloc[3]
            
            historical_games = player1_data.iloc[0:3]
            expected_ops = (
                (historical_games['ops'] * historical_games['pa']).sum() / 
                historical_games['pa'].sum()
            )
            
            assert abs(fourth_game['ops_rolling'] - expected_ops) < 0.001, \
                f"Fourth game OPS rolling should be weighted average of games 1-3. " \
                f"Expected: {expected_ops}, Got: {fourth_game['ops_rolling']}"

    def test_rolling_window_weighted_vs_simple_averages(self, batting_features, temporal_batting_data):
        """
        Test that PA-dependent metrics use weighted averages while 
        rate metrics use simple averages.
        """
        data = self._load_batting_data(2024)
        player2_data = data[data['player_id'] == 'player2'].copy()
        player2_data['game_date'] = pd.to_datetime(player2_data['game_date'])
        player2_data = player2_data.sort_values(['game_date', 'dh'])
        
        player2_data['woba'] = 0.350  # PA-weighted metric
        player2_data['babip'] = 0.300  # PA-weighted metric
        player2_data['bb_k'] = 0.50   # PA-weighted metric
        player2_data['barrel_percent'] = 8.5  # Simple average metric
        player2_data['hard_hit'] = 40.0       # Simple average metric
        player2_data['ev'] = 88.5             # PA-weighted metric
        player2_data['iso'] = 0.200           # PA-weighted metric
        player2_data['gb_fb'] = 1.20          # Simple average metric
        player2_data['baserunning'] = 0.5     # Simple average metric
        player2_data['wraa'] = 2.1            # Simple average metric
        player2_data['wpa'] = 0.15            # Simple average metric
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player2_data)
        
        window_5_stats = rolling_stats[rolling_stats['window_size'] == 5].copy()
        window_5_stats = window_5_stats.sort_values(['game_date', 'dh'])
        
        if len(window_5_stats) >= 6:
            last_game = window_5_stats.iloc[5]
            historical_games = player2_data.iloc[0:5]
            
            expected_ops_weighted = (
                (historical_games['ops'] * historical_games['pa']).sum() / 
                historical_games['pa'].sum()
            )
            assert abs(last_game['ops_rolling'] - expected_ops_weighted) < 0.001, \
                "OPS should use weighted average"
            
            expected_barrel_simple = historical_games['barrel_percent'].mean()
            assert abs(last_game['barrel_percent_rolling'] - expected_barrel_simple) < 0.001, \
                "Barrel percent should use simple average"

    def test_rolling_window_multiple_windows(self, batting_features, temporal_batting_data):
        """Test that all rolling windows (25, 14, 9, 5, 3) are calculated."""
        data = self._load_batting_data(2024)
        player1_data = data[data['player_id'] == 'player1'].copy()
        player1_data['game_date'] = pd.to_datetime(player1_data['game_date'])
        
        for metric in batting_features.rolling_metrics:
            if metric not in player1_data.columns:
                player1_data[metric] = np.random.uniform(0.1, 1.0, len(player1_data))
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player1_data)
        
        unique_windows = rolling_stats['window_size'].unique()
        expected_windows = [25, 14, 9, 5, 3]
        assert set(unique_windows) == set(expected_windows), \
            f"Expected windows {expected_windows}, got {unique_windows}"
        
        for window in expected_windows:
            window_data = rolling_stats[rolling_stats['window_size'] == window]
            assert len(window_data) == len(player1_data), \
                f"Window {window} should have {len(player1_data)} rows, got {len(window_data)}"

    def test_doubleheader_handling(self, batting_features, doubleheader_batting_data):
        """Test proper handling of doubleheader games in rolling calculations."""
        data = self._load_batting_data(2024)
        player1_data = data[data['player_id'] == 'player1'].copy()
        player1_data['game_date'] = pd.to_datetime(player1_data['game_date'])
        
        for metric in batting_features.rolling_metrics:
            if metric not in player1_data.columns:
                player1_data[metric] = 0.5
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player1_data)
        
        window_3_stats = rolling_stats[rolling_stats['window_size'] == 3]
        assert len(window_3_stats) == 3, "Should handle doubleheader games correctly"
        
        sorted_data = window_3_stats.sort_values(['game_date', 'dh'])
        assert sorted_data.iloc[0]['dh'] == 0, "First game should be dh=0"
        assert sorted_data.iloc[1]['dh'] == 1, "Second game should be dh=1"
        assert sorted_data.iloc[2]['dh'] == 0, "Third game should be dh=0"

    def test_process_player_batch(self, batting_features, temporal_batting_data):
        """Test batch processing of multiple players."""
        player_ids = ['player1', 'player2']
        
        from data.database import get_database_manager
        db = get_database_manager()
        
        for metric in ['woba', 'babip', 'bb_k', 'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa']:
            update_query = f"UPDATE batting_stats SET {metric} = ? WHERE season = ?"
            db.execute_write_query(update_query, (0.5, 2024))
        
        batch_result = batting_features._process_player_batch(player_ids)
        
        assert_dataframe_not_empty(batch_result)
        
        unique_players = batch_result['player_id'].unique()
        assert set(unique_players) == set(player_ids), \
            f"Expected players {player_ids}, got {unique_players}"
        
        unique_windows = batch_result['window_size'].unique()
        assert set(unique_windows) == set(batting_features.rolling_windows)

    def test_empty_player_batch(self, batting_features):
        """Test batch processing with no players returns empty DataFrame."""
        result = batting_features._process_player_batch([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_nonexistent_player_batch(self, batting_features, temporal_batting_data):
        """Test batch processing with nonexistent players."""
        result = batting_features._process_player_batch(['nonexistent_player'])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('data.features.player_features.batting.BATTING_CACHE_PATH', 'test_batting_{}.parquet')
    def test_caching_functionality(self, batting_features, temporal_batting_data, tmp_path):
        """Test that rolling stats are properly cached and retrieved."""
        
        from data.database import get_database_manager
        db = get_database_manager()
        
        for metric in ['woba', 'babip', 'bb_k', 'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa']:
            update_query = f"UPDATE batting_stats SET {metric} = ? WHERE season = ?"
            db.execute_write_query(update_query, (0.5, 2024))
        
        with patch('data.features.player_features.batting.FEATURES_CACHE_PATH', tmp_path):
            
            result1 = batting_features.calculate_all_player_rolling_stats()
            assert_dataframe_not_empty(result1)
            
            cache_file = tmp_path / 'test_batting_2024.parquet'
            assert cache_file.exists(), "Cache file should be created"
            
            result2 = batting_features.calculate_all_player_rolling_stats()
            
            pd.testing.assert_frame_equal(result1, result2)

    def test_rolling_stats_shift_validation(self, batting_features, temporal_batting_data):
        """
        Test that rolling calculations are properly shifted to prevent data leakage.
        Current game stats should never appear in rolling calculations for the same game.
        """
        data = self._load_batting_data(2024)
        player1_data = data[data['player_id'] == 'player1'].copy()
        player1_data['game_date'] = pd.to_datetime(player1_data['game_date'])
        player1_data = player1_data.sort_values(['game_date', 'dh'])
        
        for metric in batting_features.rolling_metrics:
            if metric not in player1_data.columns:
                player1_data[metric] = np.random.uniform(0.1, 1.0, len(player1_data))
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player1_data)
        
        window_3_stats = rolling_stats[rolling_stats['window_size'] == 3].copy()
        window_3_stats = window_3_stats.sort_values(['game_date', 'dh'])
        
        for i, row in window_3_stats.iterrows():
            current_game_ops = row['ops']
            rolling_ops = row['ops_rolling']
            
            if not pd.isna(rolling_ops):

                assert abs(rolling_ops - current_game_ops) > 0.001 or pd.isna(rolling_ops), \
                    f"Rolling OPS should not equal current game OPS for game {i}. " \
                    f"Current: {current_game_ops}, Rolling: {rolling_ops}"

    def test_window_size_metadata(self, batting_features, temporal_batting_data):
        """Test that window size and games count metadata is correctly calculated."""
        data = self._load_batting_data(2024)
        player1_data = data[data['player_id'] == 'player1'].copy()
        player1_data['game_date'] = pd.to_datetime(player1_data['game_date'])

        
        for metric in batting_features.rolling_metrics:
            if metric not in player1_data.columns:
                player1_data[metric] = 0.5
        
        rolling_stats = batting_features._calculate_rolling_window_for_player(player1_data)
        
        window_5_stats = rolling_stats[rolling_stats['window_size'] == 5].copy()
        window_5_stats = window_5_stats.sort_values(['game_date', 'dh'])

        window_5_stats = window_5_stats.reset_index()
        for i, row in window_5_stats.iterrows():
            expected_games = min(i, 5) if i > 0 else np.nan
            
            if not pd.isna(expected_games):
                assert abs(row['games_in_window'] - expected_games) < 0.1 or pd.isna(row['games_in_window']), \
                    f"Games in window should be {expected_games}, got {row['games_in_window']}"

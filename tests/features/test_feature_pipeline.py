"""
Tests for FeaturePipeline class.

Tests the feature engineering pipeline including:
- Schedule data transformation and temporal validation
- Batting features integration with lineup data
- Opponent feature addition without data leakage
- Schedule-to-odds matching with datetime reconciliation
- Complete pipeline execution and validation
- Temporal ordering and doubleheader handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch

from src.data.features.feature_pipeline import FeaturePipeline
from src.data.features.game_features.odds import Odds
from tests.conftest import (
    insert_schedule_games, insert_odds_data, insert_batting_stats, 
    insert_players, insert_lineups, insert_lineup_players,
    assert_dataframe_schema, assert_dataframe_not_empty
)


class TestFeaturePipeline:
    """Test suite for FeaturePipeline class."""

    @pytest.fixture
    def feature_pipeline(self, clean_db):
        """Create a FeaturePipeline instance."""
        return FeaturePipeline(season=2024)

    @pytest.fixture
    def sample_schedule_data(self, clean_db):
        """Sample schedule data for testing."""
        games = [
            # Regular games
            ('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS', 
             'Final', 5, 3, 'NYY', 'BOS', 0),
            ('game2', '2024-04-02', '2024-04-02T19:10:00', 2024, 'TB', 'NYY', 
             'Final', 2, 7, 'NYY', 'TB', 0),
            # Doubleheader
            ('game3a', '2024-04-03', '2024-04-03T13:05:00', 2024, 'BOS', 'TB', 
             'Final', 4, 6, 'TB', 'BOS', 0),
            ('game3b', '2024-04-03', '2024-04-03T19:05:00', 2024, 'BOS', 'TB', 
             'Final', 2, 8, 'TB', 'BOS', 1),
        ]
        
        insert_schedule_games(clean_db, games)
        
        from src.data.database import get_database_manager
        db = get_database_manager()
        
        starters = [
            ('Cole, G', 'Whitlock, K', 'game1'),
            ('Glasnow, T', 'Cortes, N', 'game2'), 
            ('Houck, T', 'Eflin, Z', 'game3a'),
            ('Pivetta, N', 'Fairbanks, P', 'game3b'),
        ]
        
        for away_starter, home_starter, game_id in starters:
            update_query = """
            UPDATE schedule 
            SET away_starter_normalized = ?, home_starter_normalized = ?
            WHERE game_id = ?
            """
            db.execute_write_query(update_query, (away_starter, home_starter, game_id))
        
        return games

    @pytest.fixture
    def sample_odds_data(self, clean_db):
        """Sample odds data for testing."""
        odds = [
            # Matching odds for games
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole, G', 'Whitlock, K',
             'DraftKings', -150, 130, -145, 125, 2024),
            ('2024-04-02', '2024-04-02T19:10:00', 'TB', 'NYY', 'Glasnow, T', 'Cortes, N',
             'DraftKings', 110, -120, 115, -125, 2024),
            # Doubleheader odds - only for first game
            ('2024-04-03', '2024-04-03T13:05:00', 'BOS', 'TB', 'Houck, T', 'Eflin, Z',
             'DraftKings', 105, -115, 100, -110, 2024),
            # Unmatched odds (different datetime)
            ('2024-04-04', '2024-04-04T20:00:00', 'LAD', 'SF', 'Kershaw, C', 'Webb, L',
             'DraftKings', -130, 110, -135, 115, 2024),
        ]
        
        insert_odds_data(clean_db, odds)
        return odds

    @pytest.fixture
    def sample_players_and_stats(self, clean_db):
        """Sample players and batting stats."""
        players = [
            (1, 'Aaron Judge', 'OF', 'NYY'),
            (2, 'Rafael Devers', 'IF', 'BOS'),
            (3, 'Wander Franco', 'IF', 'TB'),
            (4, 'Brandon Lowe', 'IF', 'TB'),
        ]
        insert_players(clean_db, players)
        
        base_date = date(2024, 3, 25)
        stats = []
        
        for player_id in ['player1', 'player2', 'player3', 'player4']:
            for i in range(7):
                game_date = base_date + timedelta(days=i)
                stats.append((
                    player_id,
                    game_date.strftime('%Y-%m-%d'),
                    'NYY' if player_id == 'player1' else ('BOS' if player_id == 'player2' else 'TB'),
                    0,  # dh
                    4,  # ab
                    4,  # pa
                    0.800 + (i * 0.02),  # ops (improving over time)
                    120 + (i * 2),  # wrc_plus
                    2024
                ))
        
        insert_batting_stats(clean_db, stats)
        return players, stats

    @pytest.fixture
    def sample_lineups(self, clean_db, sample_schedule_data, sample_players_and_stats):
        """Sample lineup data."""
        lineups = [
            # Game 1: NYY vs BOS
            ('2024-04-01', 1, 'NYY', 0, 2, 'BOS', 'player1', 2024),
            ('2024-04-01', 2, 'BOS', 0, 1, 'NYY', 'player2', 2024),
            # Game 2: TB vs NYY  
            ('2024-04-02', 3, 'TBR', 0, 1, 'NYY', 'player3', 2024),
            ('2024-04-02', 1, 'NYY', 0, 3, 'TBR', 'player1', 2024),
            # Game 3a: BOS vs TB (first game of doubleheader)
            ('2024-04-03', 2, 'BOS', 1, 3, 'TBR', 'player2', 2024),
            ('2024-04-03', 3, 'TBR', 1, 2, 'BOS', 'player3', 2024),
        ]
        
        insert_lineups(clean_db, lineups)
        
        lineup_players = [
            # Game 1 lineups
            ('2024-04-01', 1, 'NYY', 2, 'BOS', 0, 'player1', 'OF', 3, 2024),
            ('2024-04-01', 2, 'BOS', 1, 'NYY', 0, 'player2', 'IF', 4, 2024),
            # Game 2 lineups
            ('2024-04-02', 3, 'TBR', 1, 'NYY', 0, 'player3', 'IF', 2, 2024),
            ('2024-04-02', 1, 'NYY', 3, 'TBR', 0, 'player1', 'OF', 3, 2024),
            # Game 3a lineups
            ('2024-04-03', 2, 'BOS', 3, 'TBR', 0, 'player2', 'IF', 4, 2024),
            ('2024-04-03', 3, 'TBR', 2, 'BOS', 0, 'player3', 'IF', 2, 2024),
            ('2024-04-03', 3, 'TBR', 2, 'BOS', 0, 'player4', 'IF', 5, 2024),
        ]
        
        insert_lineup_players(clean_db, lineup_players)
        return lineups, lineup_players

    def test_init(self):
        """Test FeaturePipeline initialization."""
        pipeline = FeaturePipeline(season=2024)
        assert pipeline.season == 2024
        assert hasattr(pipeline, 'cache')
        assert isinstance(pipeline.cache, dict)

    def test_load_schedule_data(self, feature_pipeline, sample_schedule_data):
        """Test loading schedule data."""
        schedule_data = feature_pipeline._load_schedule_data()
        
        assert_dataframe_not_empty(schedule_data)
        expected_columns = ['game_id', 'game_date', 'away_team', 'home_team', 'dh']
        assert_dataframe_schema(schedule_data, expected_columns)
        
        assert len(schedule_data) == 4

    def test_load_odds_data(self, feature_pipeline, sample_odds_data):
        """Test loading odds data."""
        odds_data = feature_pipeline._load_odds_data()
        
        assert_dataframe_not_empty(odds_data)
        expected_columns = ['game_date', 'away_team', 'home_team', 'away_opening_odds', 'home_opening_odds']
        assert_dataframe_schema(odds_data, expected_columns)

    def test_load_batting_data(self, feature_pipeline, sample_players_and_stats):
        """Test loading batting data."""
        batting_data = feature_pipeline._load_batting_data()
        
        assert_dataframe_not_empty(batting_data)
        expected_columns = ['player_id', 'game_date', 'team', 'dh', 'pa', 'ops']
        assert_dataframe_schema(batting_data, expected_columns)

    def test_load_lineups_data(self, feature_pipeline, sample_lineups):
        """Test loading lineups data."""
        lineups_data = feature_pipeline._load_lineups_data()
        
        assert_dataframe_not_empty(lineups_data)
        expected_columns = ['game_date', 'team', 'opposing_team', 'dh', 'player_id', 'position']
        assert_dataframe_schema(lineups_data, expected_columns)

    def test_transform_schedule(self, feature_pipeline, sample_schedule_data):
        """Test schedule transformation to team perspective."""
        schedule_data = feature_pipeline._load_schedule_data()
        transformed = feature_pipeline._transform_schedule(schedule_data)
        print(transformed.index)
        print(transformed.columns)
        
        assert_dataframe_not_empty(transformed)
        
        assert len(transformed) == len(schedule_data) * 2
        
        assert 'team' in transformed.index.names
        assert 'opposing_team' in transformed.columns
        assert 'is_home' in transformed.columns
        assert 'is_winner' in transformed.columns
        
        transformed_reset = transformed.reset_index()
        
        for _, game in schedule_data.iterrows():
            game_id = game['game_id']
            game_rows = transformed_reset[transformed_reset['game_id'] == game_id]
            assert len(game_rows) == 2, f"Game {game_id} should appear twice"
            
            home_count = (game_rows['is_home'] == True).sum()
            away_count = (game_rows['is_home'] == False).sum()
            assert home_count == 1, f"Game {game_id} should have one home team"
            assert away_count == 1, f"Game {game_id} should have one away team"

    def test_transform_schedule_winner_logic(self, feature_pipeline, sample_schedule_data):
        """Test that winner logic is correctly applied in schedule transformation."""
        schedule_data = feature_pipeline._load_schedule_data()
        transformed = feature_pipeline._transform_schedule(schedule_data)
        transformed_reset = transformed.reset_index()

        game1_rows = transformed_reset[transformed_reset['game_id'] == 'game1']
        
        nyy_row = game1_rows[game1_rows['team'] == 'NYY']
        assert len(nyy_row) == 1
        assert nyy_row.iloc[0]['is_winner'] == 1, "NYY should be marked as winner"
        
        bos_row = game1_rows[game1_rows['team'] == 'BOS']
        assert len(bos_row) == 1
        assert bos_row.iloc[0]['is_winner'] == 0, "BOS should be marked as loser"

    def test_merge_schedule_with_batting_features_temporal_validation(self, feature_pipeline, sample_schedule_data, sample_players_and_stats, sample_lineups):
        """
        Test temporal validation in batting features merging.
        Ensures only historical batting data is used for each game.
        """
        from src.data.database import get_database_manager
        db = get_database_manager()
        
        for metric in ['woba', 'babip', 'bb_k', 'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa']:
            update_query = f"UPDATE batting_stats SET {metric} = ? WHERE season = ?"
            db.execute_write_query(update_query, (0.5, 2024))
        
        schedule_data = feature_pipeline._load_schedule_data()
        transformed_schedule = feature_pipeline._transform_schedule(schedule_data)
        lineups_data = feature_pipeline._load_lineups_data()
        
        player_id_map = {'player1': 1, 'player2': 2, 'player3': 3, 'player4': 4}
        lineups_data = lineups_data.copy()
        lineups_data['player_id'] = lineups_data['player_id'].map(player_id_map)
        lineups_data[['team', 'opposing_team']] = lineups_data[['team', 'opposing_team']].replace('TBR', 'TB')

        batting_features_data = []
        for game_date in pd.to_datetime(['2024-04-01', '2024-04-02', '2024-04-03']):
            for player_id, team in [(1, 'NYY'), (2, 'BOS'), (3, 'TB'), (4, 'TB')]:
                batting_features_data.append({
                    'game_date': game_date,
                    'dh': 0,
                    'player_id': player_id,
                    'team': team,
                    'team_ops_season': 0.800,
                    'team_ops_ewm_h3': 0.810,
                    'team_ops_ewm_h10': 0.820,
                    'team_ops_ewm_h25': 0.830,
                    'team_wrc_plus_season': 120,
                    'team_wrc_plus_ewm_h3': 121,
                    'team_wrc_plus_ewm_h10': 122,
                    'team_wrc_plus_ewm_h25': 123,
                    'team_frv_per_9': 0.0,
                })

        batting_features_df = pd.DataFrame(batting_features_data)

        team_features = feature_pipeline._merge_schedule_with_batting_features(
            transformed_schedule, lineups_data, batting_features_df
        )

        assert_dataframe_not_empty(team_features)

        expected_cols = [
            'team_ops_season',
            'team_ops_ewm_h3',
            'team_ops_ewm_h10',
            'team_ops_ewm_h25',
            'team_wrc_plus_season',
            'team_frv_per_9',
        ]
        assert_dataframe_schema(team_features, expected_cols)

    def test_add_opponent_features(self, feature_pipeline):
        """Test adding opponent features without data leakage."""
        test_data = pd.DataFrame({
            'feature1': [10, 20, 30, 40],
            'feature2': [0.5, 0.6, 0.7, 0.8],
        })

        index_data = [
            ('2024-04-01', 0, 'NYY', 'BOS'),
            ('2024-04-01', 0, 'BOS', 'NYY'),
            ('2024-04-02', 0, 'TB', 'NYY'),
            ('2024-04-02', 0, 'NYY', 'TB'),
        ]
        
        multi_index = pd.MultiIndex.from_tuples(
            index_data,
            names=['game_date', 'dh', 'team', 'opposing_team']
        )
        test_data.index = multi_index
        
        result = feature_pipeline._add_opponent_features(
            test_data, 
            feature_cols=['feature1', 'feature2']
        )
        
        assert_dataframe_not_empty(result)
        
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'opposing_feature1' in result.columns
        assert 'opposing_feature2' in result.columns
        
        result_reset = result.reset_index()
        
        # NYY vs BOS game
        nyy_row = result_reset[(result_reset['team'] == 'NYY') & (result_reset['opposing_team'] == 'BOS')]
        bos_row = result_reset[(result_reset['team'] == 'BOS') & (result_reset['opposing_team'] == 'NYY')]
        
        if len(nyy_row) > 0 and len(bos_row) > 0:
            assert nyy_row.iloc[0]['opposing_feature1'] == bos_row.iloc[0]['feature1']
            assert nyy_row.iloc[0]['opposing_feature2'] == bos_row.iloc[0]['feature2']

    def test_match_schedule_to_odds_basic_matching(self, feature_pipeline, sample_schedule_data, sample_odds_data):
        """Test basic schedule to odds matching."""
        schedule_data = feature_pipeline._load_schedule_data()
        odds_data = feature_pipeline._load_odds_data()
        odds_data = odds_data[odds_data['away_team'] != 'LAD']
        odds_data.loc[odds_data['game_date'] == pd.Timestamp('2024-04-03'), 'game_datetime'] = pd.Timestamp('2024-04-03T13:00:00')
        odds_features = Odds(odds_data, feature_pipeline.season).load_features()
        
        merged, raw_odds = feature_pipeline._match_schedule_to_odds(schedule_data, odds_features)
        
        assert_dataframe_not_empty(merged)
        assert_dataframe_not_empty(raw_odds)
        
        assert len(merged) == 3
        assert_dataframe_schema(
            merged,
            ['game_id', 'away_score', 'home_score', 'vig_open', 'p_open_home_median_nv']
        )

    def test_match_schedule_to_odds_datetime_reconciliation(self, feature_pipeline, clean_db):
        """Test datetime reconciliation for unmatched games."""
        games = [
            ('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS', 
             'Final', 5, 3, 'NYY', 'BOS', 0),
        ]

        insert_schedule_games(clean_db, games)
        
        odds = [
            ('2024-04-01', '2024-04-01T19:00:00', 'NYY', 'BOS', 'Cole, G', 'Whitlock, K',
             'DraftKings', -150, 130, -145, 125, 2024),
        ]
        insert_odds_data(clean_db, odds)
        
        schedule_data = feature_pipeline._load_schedule_data()
        odds_data = Odds(feature_pipeline._load_odds_data(), feature_pipeline.season).load_features()
        
        merged, raw_odds = feature_pipeline._match_schedule_to_odds(schedule_data, odds_data)
        
        assert_dataframe_not_empty(merged)
        assert_dataframe_not_empty(raw_odds)
        assert merged.reset_index().iloc[0]['game_id'] == 'game1'

    def test_doubleheader_handling_in_pipeline(self, feature_pipeline, sample_schedule_data, sample_lineups):
        """Test that doubleheaders are properly handled throughout the pipeline."""
        schedule_data = feature_pipeline._load_schedule_data()
        transformed = feature_pipeline._transform_schedule(schedule_data)

        dh_games = transformed.reset_index()
        dh_games = dh_games[dh_games['game_date'] == '2024-04-03']
        
        assert len(dh_games) == 4, "Doubleheader should create 4 rows (2 games × 2 teams)"
        
        dh_values = dh_games['dh'].unique()
        assert 0 in dh_values, "Should have dh=0 games"
        assert 1 in dh_values, "Should have dh=1 games"

    def test_get_batting_features_integration(self, feature_pipeline, 
                                            sample_schedule_data, sample_players_and_stats, sample_lineups):
        """Test integration of batting features calculation."""
        mock_batting_features = pd.DataFrame({
            'game_date': pd.to_datetime(['2024-04-01'] * 4 + ['2024-04-02'] * 4 + ['2024-04-03'] * 4),
            'dh': [0] * 12,
            'player_id': [1, 2, 3, 4] * 3,
            'mlb_id': [1, 2, 3, 4] * 3,
            'team': ['NYY', 'BOS', 'TB', 'TB'] * 3,
            'ops_season': [0.850, 0.780, 0.760, 0.740] * 3,
            'ops_ewm_h3': [0.851, 0.781, 0.761, 0.741] * 3,
            'ops_ewm_h10': [0.852, 0.782, 0.762, 0.742] * 3,
            'ops_ewm_h25': [0.853, 0.783, 0.763, 0.743] * 3,
            'wrc_plus_season': [125, 110, 105, 100] * 3,
            'wrc_plus_ewm_h3': [126, 111, 106, 101] * 3,
            'wrc_plus_ewm_h10': [127, 112, 107, 102] * 3,
            'wrc_plus_ewm_h25': [128, 113, 108, 103] * 3,
        })

        batting_comp_cols = [
            'woba', 'hard_hit', 'barrel_percent', 'bb_k', 'babip',
            'ev', 'iso', 'baserunning', 'wpa'
        ]
        for stat_idx, stat in enumerate(batting_comp_cols, start=1):
            for suffix_idx, suffix in enumerate(['season', 'ewm_h3', 'ewm_h10', 'ewm_h25'], start=1):
                mock_batting_features[f'{stat}_{suffix}'] = 0.1 * stat_idx + 0.01 * suffix_idx

        mock_fielding_features = pd.DataFrame({
            'player_id': [1, 2, 3, 4],
            'month': [4, 4, 4, 4],
            'frv_per_9': [0.1, 0.2, 0.3, 0.4],
        })

        mock_raw_batting = pd.DataFrame({
            'player_id': [1, 2, 3, 4],
            'pos': ['OF', 'IF', 'IF', 'IF'],
        })

        mock_lineups = feature_pipeline._load_lineups_data().copy()
        mock_lineups['player_id'] = mock_lineups['player_id'].map({
            'player1': 1, 'player2': 2, 'player3': 3, 'player4': 4
        })
        mock_lineups[['team', 'opposing_team']] = mock_lineups[['team', 'opposing_team']].replace('TBR', 'TB')
        
        schedule_data = feature_pipeline._load_schedule_data()
        transformed_schedule = feature_pipeline._transform_schedule(schedule_data)
        
        with patch.object(feature_pipeline, '_load_batting_data', return_value=mock_raw_batting), \
             patch.object(feature_pipeline, '_load_lineups_data', return_value=mock_lineups), \
             patch.object(feature_pipeline, '_load_fielding_data', return_value=pd.DataFrame()), \
             patch('src.data.features.feature_pipeline.BattingFeatures.load_features', return_value=mock_batting_features) as mock_batting_load, \
             patch('src.data.features.feature_pipeline.FieldingFeatures.load_features', return_value=mock_fielding_features) as mock_fielding_load:
            team_features = feature_pipeline._get_batting_features(transformed_schedule)

        mock_batting_load.assert_called_once()
        mock_fielding_load.assert_called_once()
        
        assert isinstance(team_features, pd.DataFrame)
        assert_dataframe_not_empty(team_features)
        assert 'opposing_team_ops_season' in team_features.columns
        assert 'ops_season_diff' in team_features.columns

    def test_temporal_ordering_validation(self, feature_pipeline, sample_schedule_data):
        """Test that all data maintains proper temporal ordering."""
        schedule_data = feature_pipeline._load_schedule_data()
        transformed = feature_pipeline._transform_schedule(schedule_data)
        
        transformed_reset = transformed.reset_index()
        transformed_reset['game_date'] = pd.to_datetime(transformed_reset['game_date'])
        
        prev_date = None
        prev_dh = None
        
        for _, row in transformed_reset.iterrows():
            current_date = row['game_date']
            current_dh = row['dh']
            
            if prev_date is not None:
                assert current_date >= prev_date, \
                    f"Dates should be in order: {prev_date} <= {current_date}"
                
                if current_date == prev_date and prev_dh is not None:
                    assert current_dh >= prev_dh, \
                        f"DH should be in order for same date: {prev_dh} <= {current_dh}"
            
            prev_date = current_date
            prev_dh = current_dh

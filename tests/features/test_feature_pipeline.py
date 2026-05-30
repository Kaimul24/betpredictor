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

from src.data.features.base_feature import BaseFeatures
from src.data.features.feature_pipeline import FeaturePipeline
from src.data.features.game_features.odds import Odds
from tests.conftest import (
    insert_schedule_games, insert_odds_data, insert_batting_stats, 
    insert_players, insert_lineups, insert_lineup_players,
    assert_dataframe_schema, assert_dataframe_not_empty
)


class TestFeaturePipeline:
    """Test suite for FeaturePipeline class."""

    TEAM_METRIC_COLS = [
        'win_pct',
        'pyth_expectation',
        'run_diff',
        'one_run_win_pct',
    ]
    TEAM_METRIC_SUFFIXES = ['season', 'ewm_h3', 'ewm_h8', 'ewm_h20']
    STARTER_COLS = [
        'starter_era',
        'starter_babip',
        'starter_hard_hit',
        'starter_k_percent',
        'starter_barrel_percent',
        'starter_fip',
        'starter_siera',
        'starter_stuff',
        'starter_ev',
        'starter_hr_fb',
        'starter_wpa',
        'starter_bb_percent',
    ]
    PEN_COLS = [
        'pen_era',
        'pen_babip',
        'pen_hard_hit',
        'pen_k_percent',
        'pen_barrel_percent',
        'pen_fip',
        'pen_siera',
        'pen_stuff',
        'pen_ev',
        'pen_hr_fb',
        'pen_wpa_li',
    ]
    PITCHING_SUFFIXES = ['season', 'ewm_h3', 'ewm_h8', 'ewm_h20']

    @staticmethod
    def _pipeline_schedule(include_far_future=False):
        rows = [
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'game_datetime': pd.Timestamp('2024-04-01T19:05:00'),
                'day_night_game': 'night',
                'season': 2024,
                'away_team': 'NYY',
                'home_team': 'BOS',
                'dh': 0,
                'venue_name': 'Fenway Park',
                'venue_id': 1,
                'venue_elevation': 20,
                'venue_timezone': 'America/New_York',
                'venue_gametime_offset': -4,
                'status': 'Final',
                'away_probable_pitcher': 'Cole, G',
                'home_probable_pitcher': 'Whitlock, K',
                'away_starter_normalized': 'Cole, G',
                'home_starter_normalized': 'Whitlock, K',
                'away_pitcher_id': 11,
                'home_pitcher_id': 22,
                'wind': '8 mph, Out To CF',
                'condition': 'Clear',
                'temp': 72,
                'away_score': 5,
                'home_score': 3,
                'winning_team': 'NYY',
                'losing_team': 'BOS',
            }
        ]

        if include_far_future:
            far_future = rows[0].copy()
            far_future.update({
                'game_id': 'game_far',
                'game_date': pd.Timestamp('2024-04-02'),
                'game_datetime': pd.Timestamp('2024-04-06T19:05:00'),
                'away_team': 'TB',
                'home_team': 'BAL',
                'winning_team': 'BAL',
                'losing_team': 'TB',
            })
            rows.append(far_future)

        return pd.DataFrame(rows)

    @staticmethod
    def _odds_matched_schedule(schedule_df):
        matched = schedule_df.copy()
        matched['vig_open'] = 1.05
        matched['p_open_home_median_nv'] = 0.54
        matched['p_open_home_mean_nv'] = 0.54
        matched['p_open_home_std_nv'] = 0.0
        matched['p_open_away_median_nv'] = 0.46
        matched['p_open_away_mean_nv'] = 0.46
        matched['p_open_away_std_nv'] = 0.0
        matched['num_books'] = 1
        matched['logit_prob_home_std_nv'] = 0.0
        return matched

    @staticmethod
    def _team_indexed(rows):
        return pd.DataFrame(rows).set_index([
            'game_id',
            'game_date',
            'dh',
            'game_datetime',
            'team',
            'opposing_team',
        ])

    def _batting_provider_output(self):
        rows = [
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'team': 'NYY',
                'opposing_team': 'BOS',
            },
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'team': 'BOS',
                'opposing_team': 'NYY',
            },
        ]

        df = pd.DataFrame(rows)
        batting_cols = [
            'woba', 'wrc_plus', 'hard_hit', 'barrel_percent', 'bb_k', 'ops',
            'babip', 'ev', 'iso', 'baserunning', 'wpa', 'k_percent', 'bb_percent',
        ]
        extra_cols = {}
        for stat_idx, stat in enumerate(batting_cols, start=1):
            for suffix_idx, suffix in enumerate(['season', 'ewm_h3', 'ewm_h10', 'ewm_h25'], start=1):
                extra_cols[f'team_{stat}_{suffix}'] = stat_idx + suffix_idx / 10.0
                extra_cols[f'opposing_team_{stat}_{suffix}'] = stat_idx + suffix_idx / 10.0 + 1.0

        extra_cols['team_frv_per_9'] = [0.1, 0.2]
        extra_cols['opposing_team_frv_per_9'] = [0.2, 0.1]
        df = pd.concat([df, pd.DataFrame(extra_cols)], axis=1)
        df['ops_season_diff'] = df['team_ops_season'] - df['opposing_team_ops_season']
        return df

    def _team_features_provider_output(self):
        rows = [
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'game_datetime': pd.Timestamp('2024-04-01T19:05:00'),
                'team': 'NYY',
                'opposing_team': 'BOS',
            },
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'game_datetime': pd.Timestamp('2024-04-01T19:05:00'),
                'team': 'BOS',
                'opposing_team': 'NYY',
            },
        ]
        df = self._team_indexed(rows)

        for stat_idx, stat in enumerate(self.TEAM_METRIC_COLS, start=1):
            for suffix_idx, suffix in enumerate(self.TEAM_METRIC_SUFFIXES, start=1):
                df[f'{stat}_{suffix}'] = stat_idx + suffix_idx / 10.0

        return df

    def _pitching_provider_output(self):
        rows = [
            {
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'season': 2024,
                'team': 'NYY',
                'opposing_team': 'BOS',
                'team_starter_name': 'Cole, G',
                'opposing_team_starter_name': 'Whitlock, K',
            },
            {
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'season': 2024,
                'team': 'BOS',
                'opposing_team': 'NYY',
                'team_starter_name': 'Whitlock, K',
                'opposing_team_starter_name': 'Cole, G',
            },
        ]
        df = pd.DataFrame(rows)

        extra_cols = {}
        for prefix in ['team', 'opposing_team']:
            for stat_idx, stat in enumerate(self.STARTER_COLS, start=1):
                for suffix_idx, suffix in enumerate(self.PITCHING_SUFFIXES, start=1):
                    extra_cols[f'{prefix}_{stat}_{suffix}'] = stat_idx + suffix_idx / 10.0

        for stat_idx, stat in enumerate(self.PEN_COLS, start=1):
            for suffix_idx, suffix in enumerate(self.PITCHING_SUFFIXES, start=1):
                extra_cols[f'team_{stat}_{suffix}'] = stat_idx + suffix_idx / 10.0

        extra_cols['team_pen_rest_days_mean'] = [2.0, 3.0]
        extra_cols['team_pen_rest_days_median'] = [2.0, 3.0]
        extra_cols['team_pen_freshness_mean'] = [0.5, 0.8]
        extra_cols['team_pen_freshness_gmliw'] = [0.4, 0.7]
        extra_cols['team_pen_hi_lev_available'] = [0.9, 0.6]
        extra_cols['team_starter_last_app_date'] = pd.Timestamp('2024-03-25')
        extra_cols['opposing_team_starter_last_app_date'] = pd.Timestamp('2024-03-25')
        df = pd.concat([df, pd.DataFrame(extra_cols)], axis=1)
        return df

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
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole, G', 'Bello, B',
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
        
        for player_id in [1, 2, 3, 4]:
            for i in range(7):
                game_date = base_date + timedelta(days=i)
                stats.append((
                    player_id,
                    game_date.strftime('%Y-%m-%d'),
                    'NYY' if player_id == 1 else ('BOS' if player_id == 2 else 'TB'),
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
            ('2024-04-01', 1, 'NYY', 0, 2, 'BOS', 1, 2024),
            ('2024-04-01', 2, 'BOS', 0, 1, 'NYY', 2, 2024),
            # Game 2: TB vs NYY  
            ('2024-04-02', 3, 'TBR', 0, 1, 'NYY', 3, 2024),
            ('2024-04-02', 1, 'NYY', 0, 3, 'TBR', 1, 2024),
            # Game 3a: BOS vs TB (first game of doubleheader)
            ('2024-04-03', 2, 'BOS', 1, 3, 'TBR', 2, 2024),
            ('2024-04-03', 3, 'TBR', 1, 2, 'BOS', 3, 2024),
        ]
        
        insert_lineups(clean_db, lineups)
        
        lineup_players = [
            # Game 1 lineups
            ('2024-04-01', 1, 'NYY', 2, 'BOS', 0, 1, 'OF', 3, 2024),
            ('2024-04-01', 2, 'BOS', 1, 'NYY', 0, 2, 'IF', 4, 2024),
            # Game 2 lineups
            ('2024-04-02', 3, 'TBR', 1, 'NYY', 0, 3, 'IF', 2, 2024),
            ('2024-04-02', 1, 'NYY', 3, 'TBR', 0, 1, 'OF', 3, 2024),
            # Game 3a lineups
            ('2024-04-03', 2, 'BOS', 3, 'TBR', 0, 2, 'IF', 4, 2024),
            ('2024-04-03', 3, 'TBR', 2, 'BOS', 0, 3, 'IF', 2, 2024),
            ('2024-04-03', 3, 'TBR', 2, 'BOS', 0, 4, 'IF', 5, 2024),
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

    def test_merge_schedule_with_position_player_features_aggregates_home_away_lineups(self, feature_pipeline, sample_schedule_data, sample_players_and_stats, sample_lineups):
        """
        Test matching game-level schedule rows to home/away lineups and averaging player features by side.
        """
        from src.data.database import get_database_manager
        db = get_database_manager()
        
        for metric in ['woba', 'babip', 'bb_k', 'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa']:
            update_query = f"UPDATE batting_stats SET {metric} = ? WHERE season = ?"
            db.execute_write_query(update_query, (0.5, 2024))
        
        schedule_data = feature_pipeline._load_schedule_data()
        lineups_data = feature_pipeline._load_lineups_data()
        
        lineups_data = lineups_data.copy()
        lineups_data['game_date'] = pd.to_datetime(lineups_data['game_date'])
        lineups_data[['team', 'opposing_team']] = lineups_data[['team', 'opposing_team']].replace('TBR', 'TB')

        batting_features_data = []
        for game_date in pd.to_datetime(['2024-04-01', '2024-04-02', '2024-04-03']):
            for player_id, team in [(1, 'NYY'), (2, 'BOS'), (3, 'TB'), (4, 'TB')]:
                batting_features_data.append({
                    'game_date': game_date,
                    'dh': 0,
                    'player_id': player_id,
                    'mlb_id': player_id,
                    'team': team,
                    'season': 2024,
                    'ops_season': 0.800 + (player_id / 100),
                    'ops_ewm_h3': 0.810 + (player_id / 100),
                    'ops_ewm_h10': 0.820 + (player_id / 100),
                    'ops_ewm_h25': 0.830 + (player_id / 100),
                    'wrc_plus_season': 120 + player_id,
                    'wrc_plus_ewm_h3': 121 + player_id,
                    'wrc_plus_ewm_h10': 122 + player_id,
                    'wrc_plus_ewm_h25': 123 + player_id,
                    'frv_per_9': player_id / 10,
                })

        batting_features_df = pd.DataFrame(batting_features_data)

        team_features = feature_pipeline._merge_schedule_with_position_player_features(
            schedule_data, lineups_data, batting_features_df
        )

        assert_dataframe_not_empty(team_features)

        expected_cols = [
            'away_ops_season',
            'away_ops_ewm_h3',
            'away_ops_ewm_h10',
            'away_ops_ewm_h25',
            'away_wrc_plus_season',
            'away_frv_per_9',
            'home_ops_season',
            'home_ops_ewm_h3',
            'home_ops_ewm_h10',
            'home_ops_ewm_h25',
            'home_wrc_plus_season',
            'home_frv_per_9',
        ]
        assert_dataframe_schema(team_features, expected_cols)

        game1 = team_features.loc[('game1', pd.Timestamp('2024-04-01'), 'BOS', 'NYY', 0)]
        assert game1['away_ops_season'] == pytest.approx(0.810)
        assert game1['home_ops_season'] == pytest.approx(0.820)

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
        """Doubleheaders stay as distinct game-level rows with preserved dh values."""
        schedule_data = feature_pipeline._load_schedule_data()
        dh_games = schedule_data[schedule_data['game_date'] == pd.Timestamp('2024-04-03')]

        assert len(dh_games) == 2
        assert dh_games['dh'].tolist() == [0, 1]
        assert dh_games['away_team'].tolist() == ['BOS', 'BOS']
        assert dh_games['home_team'].tolist() == ['TB', 'TB']
        assert dh_games['game_id'].tolist() == ['game3a', 'game3b']
        assert dh_games['game_datetime'].nunique() == 2

        lineups_data = feature_pipeline._load_lineups_data().copy()
        lineups_data['game_date'] = pd.to_datetime(lineups_data['game_date'])
        lineups_data[['team', 'opposing_team']] = lineups_data[['team', 'opposing_team']].replace('TBR', 'TB')

        batting_features = pd.DataFrame({
            'game_date': pd.to_datetime(['2024-04-03'] * 2),
            'dh': [0, 0],
            'player_id': [2, 3],
            'mlb_id': [2, 3],
            'team': ['BOS', 'TB'],
            'season': [2024, 2024],
            'ops_season': [0.780, 0.760],
            'ops_ewm_h3': [0.790, 0.770],
            'frv_per_9': [0.2, 0.3],
        })

        result = feature_pipeline._merge_schedule_with_position_player_features(
            dh_games, lineups_data, batting_features
        )

        assert result.index.get_level_values('dh').tolist() == [0, 1]

    def test_get_position_player_features_integration(self, feature_pipeline,
                                                      sample_schedule_data, sample_players_and_stats, sample_lineups):
        """Position-player features are produced from game-level schedules with home/away columns."""
        mock_batting_features = pd.DataFrame({
            'game_date': pd.to_datetime(['2024-04-01'] * 4 + ['2024-04-02'] * 4 + ['2024-04-03'] * 4),
            'dh': [0] * 12,
            'player_id': [1, 2, 3, 4] * 3,
            'mlb_id': [1, 2, 3, 4] * 3,
            'team': ['NYY', 'BOS', 'TB', 'TB'] * 3,
            'season': [2024] * 12,
        })

        batting_comp_cols = [
            'woba', 'wrc_plus', 'hard_hit', 'barrel_percent', 'bb_k', 'ops',
            'babip', 'ev', 'iso', 'baserunning', 'wpa',
        ]
        for stat_idx, stat in enumerate(batting_comp_cols, start=1):
            for suffix_idx, suffix in enumerate(['season', 'ewm_h3', 'ewm_h10', 'ewm_h25'], start=1):
                mock_batting_features[f'{stat}_{suffix}'] = (
                    0.1 * stat_idx
                    + 0.01 * suffix_idx
                    + mock_batting_features['player_id'] / 1000
                )

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
        mock_lineups[['team', 'opposing_team']] = mock_lineups[['team', 'opposing_team']].replace('TBR', 'TB')
        schedule_data = feature_pipeline._load_schedule_data()

        with patch.object(feature_pipeline, '_load_batting_data', return_value=mock_raw_batting), \
             patch.object(feature_pipeline, '_load_lineups_data', return_value=mock_lineups), \
             patch.object(feature_pipeline, '_load_fielding_data', return_value=pd.DataFrame()), \
             patch('src.data.features.feature_pipeline.BattingFeatures.load_features', return_value=mock_batting_features) as mock_batting_load, \
             patch('src.data.features.feature_pipeline.FieldingFeatures.load_features', return_value=mock_fielding_features) as mock_fielding_load:
            team_features = feature_pipeline._get_position_player_features(schedule_data)

        mock_batting_load.assert_called_once()
        mock_fielding_load.assert_called_once()

        assert isinstance(team_features, pd.DataFrame)
        assert_dataframe_not_empty(team_features)
        assert 'home_ops_season' in team_features.columns
        assert 'away_ops_season' in team_features.columns
        assert 'home_frv_per_9' in team_features.columns
        assert 'away_frv_per_9' in team_features.columns
        assert 'home_away_ops_season_diff' in team_features.columns

    def test_merge_batting_fielding_features_uses_normalized_months(self, feature_pipeline):
        """Fielding features are monthly and March/October batting rows map to regular-season months."""
        batting_stats = pd.DataFrame({
            'player_id': [101, 102, 103],
            'mlb_id': [1, 2, 3],
            'game_date': pd.to_datetime(['2024-03-30', '2024-10-01', '2024-05-01']),
            'dh': [0, 0, 0],
            'team': ['NYY', 'BOS', 'TB'],
            'ops_season': [0.800, 0.700, 0.650],
        })
        fielding_stats = pd.DataFrame({
            'player_id': [1, 2],
            'month': [4, 9],
            'frv_per_9': [1.5, -0.5],
            'team_fld': ['drop-me', 'drop-me-too'],
        })

        merged = feature_pipeline._merge_batting_fielding_features(batting_stats, fielding_stats)

        assert merged.loc[merged['mlb_id'] == 1, 'month'].iloc[0] == 4
        assert merged.loc[merged['mlb_id'] == 1, 'frv_per_9'].iloc[0] == 1.5
        assert merged.loc[merged['mlb_id'] == 2, 'month'].iloc[0] == 9
        assert merged.loc[merged['mlb_id'] == 2, 'frv_per_9'].iloc[0] == -0.5
        assert np.isnan(merged.loc[merged['mlb_id'] == 3, 'frv_per_9'].iloc[0])
        assert not any(col.endswith('_fld') for col in merged.columns)

    def test_merge_schedule_with_position_player_features_averages_and_fills_missing_players(self, feature_pipeline):
        """Lineup player features are averaged by home/away side, with missing stats filled by league medians."""
        schedule_df = pd.DataFrame({
            'game_id': ['game1'],
            'game_date': [pd.Timestamp('2024-04-01')],
            'dh': [0],
            'game_datetime': [pd.Timestamp('2024-04-01T19:05:00')],
            'away_team': ['NYY'],
            'home_team': ['BOS'],
        })
        lineups_data = pd.DataFrame({
            'game_date': [pd.Timestamp('2024-04-01')] * 3,
            'team': ['NYY', 'NYY', 'BOS'],
            'opposing_team': ['BOS', 'BOS', 'NYY'],
            'dh': [0, 0, 0],
            'player_id': [1, 3, 2],
            'batting_order': [1, 2, 1],
            'position': ['OF', 'IF', 'IF'],
            'season': [2024, 2024, 2024],
        })
        batting_features = pd.DataFrame({
            'game_date': [pd.Timestamp('2024-04-01'), pd.Timestamp('2024-04-01')],
            'dh': [0, 0],
            'player_id': [1, 2],
            'mlb_id': [1, 2],
            'team': ['NYY', 'BOS'],
            'season': [2024, 2024],
            'ops_season': [0.800, 0.600],
            'ops_ewm_h3': [0.900, 0.700],
            'frv_per_9': [1.0, 3.0],
        })

        result = feature_pipeline._merge_schedule_with_position_player_features(
            schedule_df, lineups_data, batting_features
        )

        assert len(result) == 1
        game = result.loc[('game1', pd.Timestamp('2024-04-01'), 'BOS', 'NYY', 0)]
        assert game['away_ops_season'] == pytest.approx(0.75)
        assert game['away_ops_ewm_h3'] == pytest.approx(0.85)
        assert game['away_frv_per_9'] == pytest.approx(1.5)
        assert game['home_ops_season'] == pytest.approx(0.600)
        assert game['home_frv_per_9'] == pytest.approx(3.0)

    def test_add_opponent_features_default_columns_and_missing_reciprocal(self, feature_pipeline):
        """Opponent alignment skips existing opponent columns and leaves missing reciprocal rows null."""
        df = pd.DataFrame({
            'feature1': [10, 20, 30],
            'opposing_existing': [99, 98, 97],
        }, index=pd.MultiIndex.from_tuples(
            [
                ('2024-04-01', 0, 'NYY', 'BOS'),
                ('2024-04-01', 0, 'BOS', 'NYY'),
                ('2024-04-02', 0, 'TB', 'NYY'),
            ],
            names=['game_date', 'dh', 'team', 'opposing_team'],
        ))

        result = feature_pipeline._add_opponent_features(df)

        assert 'opposing_feature1' in result.columns
        assert 'opposing_opposing_existing' not in result.columns
        assert result.loc[('2024-04-01', 0, 'NYY', 'BOS'), 'opposing_feature1'] == 20
        assert np.isnan(result.loc[('2024-04-02', 0, 'TB', 'NYY'), 'opposing_feature1'])

    def test_add_opponent_features_rejects_duplicate_indexes(self, feature_pipeline):
        df = pd.DataFrame({
            'feature1': [10, 20],
        }, index=pd.MultiIndex.from_tuples(
            [
                ('2024-04-01', 0, 'NYY', 'BOS'),
                ('2024-04-01', 0, 'NYY', 'BOS'),
            ],
            names=['game_date', 'dh', 'team', 'opposing_team'],
        ))

        with pytest.raises(ValueError, match='duplicates'):
            feature_pipeline._add_opponent_features(df)

    def test_matchup_diff_helpers_create_expected_values(self):
        df = pd.DataFrame({
            'home_ops_season': [0.800],
            'away_ops_season': [0.700],
            'home_ops_ewm_h3': [0.820],
            'away_ops_ewm_h3': [0.760],
            'home_starter_fip_season': [3.5],
            'away_woba_season': [0.320],
            'home_starter_fip_ewm_h3': [3.2],
            'away_woba_ewm_h10': [0.300],
        })

        same_base = BaseFeatures._add_matchup_cols_diff_same_base(
            df, cols=['ops'], ewm_cols=['season', 'ewm_h3']
        )
        cross_base = BaseFeatures._add_matchup_cols_diff_base(
            df,
            col1=['starter_fip'],
            col2=['woba'],
            col1_ewm_cols=['season', 'ewm_h3'],
            col2_ewm_cols=['season', 'ewm_h10'],
        )

        assert same_base['home_away_ops_season_diff'].iloc[0] == pytest.approx(0.100)
        assert same_base['home_away_ops_ewm_h3_diff'].iloc[0] == pytest.approx(0.060)
        assert cross_base['home_away_starter_fip_woba_season_diff'].iloc[0] == pytest.approx(3.180)
        assert cross_base['home_away_starter_fip_woba_ewm_h3_diff'].iloc[0] == pytest.approx(2.900)

    def test_matchup_diff_base_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match='same length'):
            BaseFeatures._add_matchup_cols_diff_base(
                pd.DataFrame(),
                col1=['home_starter_fip', 'home_starter_k_percent'],
                col2=['woba'],
                col1_ewm_cols=['season'],
                col2_ewm_cols=['season'],
            )

    def test_handle_unmatched_games_doubleheader_uses_closest_time(self, feature_pipeline, clean_db):
        games = [
            ('game1a', '2024-04-01', '2024-04-01T13:05:00', 2024, 'NYY', 'BOS',
             'Final', 5, 3, 'NYY', 'BOS', 0),
            ('game1b', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS',
             'Final', 4, 6, 'BOS', 'NYY', 1),
        ]
        insert_schedule_games(clean_db, games)
        schedule = feature_pipeline._load_schedule_data()
        unmatched = pd.DataFrame({
            'game_date': [pd.Timestamp('2024-04-01'), pd.Timestamp('2024-04-01')],
            'game_datetime': [pd.Timestamp('2024-04-01T12:55:00'), pd.Timestamp('2024-04-01T19:15:00')],
            'away_team': ['NYY', 'NYY'],
            'home_team': ['BOS', 'BOS'],
            'sportsbook': ['BookA', 'BookB'],
            'vig_open': [1.05, 1.04],
            'p_open_home_median_nv': [0.52, 0.55],
        })

        reconciled = feature_pipeline._handle_unmatched_games(schedule, unmatched)

        assert reconciled.set_index('sportsbook').loc['BookA', 'dh'] == 0
        assert reconciled.set_index('sportsbook').loc['BookB', 'dh'] == 1
        assert 'game_datetime' not in reconciled.columns

    def test_handle_unmatched_games_omits_games_without_schedule_match(self, feature_pipeline, sample_schedule_data):
        schedule = feature_pipeline._load_schedule_data()
        unmatched = pd.DataFrame({
            'game_date': [pd.Timestamp('2024-04-20')],
            'game_datetime': [pd.Timestamp('2024-04-20T19:05:00')],
            'away_team': ['LAD'],
            'home_team': ['SF'],
            'sportsbook': ['BookA'],
        })

        assert feature_pipeline._handle_unmatched_games(schedule, unmatched).empty

    def test_match_schedule_to_odds_filters_rows_with_missing_odds_features(self, feature_pipeline, clean_db):
        games = [
            ('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS',
             'Final', 5, 3, 'NYY', 'BOS', 0),
            ('game2', '2024-04-02', '2024-04-02T19:10:00', 2024, 'TB', 'NYY',
             'Final', 2, 7, 'NYY', 'TB', 0),
        ]
        insert_schedule_games(clean_db, games)
        odds = [
            ('2024-04-01', '2024-04-01T19:05:00', 'NYY', 'BOS', 'Cole, G', 'Whitlock, K',
             'DraftKings', -150, 130, -145, 125, 2024),
            ('2024-04-02', '2024-04-02T19:00:00', 'TB', 'NYY', 'Glasnow, T', 'Cortes, N',
             'DraftKings', 110, -120, 115, -125, 2024),
        ]
        insert_odds_data(clean_db, odds)
        schedule = feature_pipeline._load_schedule_data()
        odds_features = Odds(feature_pipeline._load_odds_data(), feature_pipeline.season).load_features()
        odds_features.loc[odds_features['away_team'] == 'NYY', 'vig_open'] = np.nan

        merged, _ = feature_pipeline._match_schedule_to_odds(schedule, odds_features)

        assert merged.reset_index()['game_id'].tolist() == ['game2']

    def test_start_pipeline_builds_full_home_away_features_and_drops_far_future_games(self, feature_pipeline):
        schedule = self._pipeline_schedule(include_far_future=True)
        second_game = schedule.iloc[0].copy()
        second_game.update({
            'game_id': 'game2',
            'game_date': pd.Timestamp('2024-04-02'),
            'game_datetime': pd.Timestamp('2024-04-02T19:10:00'),
            'away_team': 'TB',
            'home_team': 'BAL',
            'away_score': 2,
            'home_score': 7,
            'winning_team': 'BAL',
            'losing_team': 'TB',
        })
        schedule = pd.concat([schedule, second_game.to_frame().T], ignore_index=True)

        raw_odds = pd.DataFrame({'sportsbook': ['DraftKings']})
        odds_features = pd.DataFrame({'sportsbook': ['DraftKings']})
        idx = ['game_id', 'game_date', 'dh', 'home_team', 'away_team']

        def indexed_feature_frame(rows):
            return pd.DataFrame(rows).set_index(idx)

        valid_schedule_seen_by_match = None

        def match_schedule_to_odds(schedule_arg, odds_arg):
            nonlocal valid_schedule_seen_by_match
            valid_schedule_seen_by_match = schedule_arg.copy()
            assert schedule_arg['game_id'].tolist() == ['game1', 'game2']
            assert odds_arg is odds_features
            return self._odds_matched_schedule(schedule_arg), raw_odds

        def rows_for_side_features(base_rows, cols, suffixes, home_offset=1.0, away_offset=0.0):
            rows = []
            for row_idx, base_row in enumerate(base_rows):
                out = {
                    'game_id': base_row['game_id'],
                    'game_date': base_row['game_date'],
                    'dh': base_row['dh'],
                    'home_team': base_row['home_team'],
                    'away_team': base_row['away_team'],
                }
                for stat_idx, stat in enumerate(cols, start=1):
                    for suffix_idx, suffix in enumerate(suffixes, start=1):
                        base_value = stat_idx * 10 + suffix_idx + row_idx / 10
                        out[f'home_{stat}_{suffix}'] = base_value + home_offset
                        out[f'away_{stat}_{suffix}'] = base_value + away_offset
                rows.append(out)
            return rows

        matched_base_rows = [
            {
                'game_id': 'game1',
                'game_date': pd.Timestamp('2024-04-01'),
                'dh': 0,
                'home_team': 'BOS',
                'away_team': 'NYY',
            },
            {
                'game_id': 'game2',
                'game_date': pd.Timestamp('2024-04-02'),
                'dh': 0,
                'home_team': 'BAL',
                'away_team': 'TB',
            },
        ]

        batting_cols = [
            'woba', 'wrc_plus', 'hard_hit', 'barrel_percent', 'bb_k', 'ops',
            'babip', 'ev', 'iso', 'baserunning', 'wpa', 'k_percent',
            'bb_percent',
        ]
        position_player_features = indexed_feature_frame(
            rows_for_side_features(
                matched_base_rows,
                batting_cols,
                ['season', 'ewm_h3', 'ewm_h10', 'ewm_h25'],
                home_offset=0.5,
                away_offset=0.1,
            )
        )

        pitching_cols = self.STARTER_COLS + self.PEN_COLS
        pitching_features = indexed_feature_frame(
            rows_for_side_features(
                matched_base_rows,
                pitching_cols,
                self.PITCHING_SUFFIXES,
                home_offset=2.0,
                away_offset=0.25,
            )
        )

        team_features = indexed_feature_frame(
            rows_for_side_features(
                matched_base_rows,
                self.TEAM_METRIC_COLS,
                self.TEAM_METRIC_SUFFIXES,
                home_offset=0.75,
                away_offset=0.25,
            )
        )

        context_features = indexed_feature_frame([
            {
                **base_row,
                'park_factor': 102 + idx_,
                'context_temp': 72 + idx_,
            }
            for idx_, base_row in enumerate(matched_base_rows)
        ])

        with patch.object(feature_pipeline, '_load_schedule_data', return_value=schedule), \
            patch.object(feature_pipeline, '_load_odds_data', return_value=pd.DataFrame({'raw': [1]})), \
            patch.object(feature_pipeline, '_match_schedule_to_odds', side_effect=match_schedule_to_odds), \
            patch.object(feature_pipeline, '_get_position_player_features', return_value=position_player_features) as mock_get_position, \
            patch.object(feature_pipeline, '_get_pitcher_features', return_value=pitching_features) as mock_get_pitcher, \
            patch('src.data.features.feature_pipeline.Odds') as mock_odds, \
            patch('src.data.features.feature_pipeline.GameContextFeatures') as mock_context_cls, \
            patch('src.data.features.feature_pipeline.TeamFeatures') as mock_team_cls:
            
            mock_odds.return_value.load_features.return_value = odds_features
            mock_context_cls.return_value.load_features.return_value = context_features
            mock_team_cls.return_value.load_features.return_value = team_features
            final_features, returned_raw_odds = feature_pipeline.start_pipeline(force_recreate=True, mkt_only=True)

        assert returned_raw_odds is raw_odds
        mock_odds.assert_called_once()
        assert mock_odds.call_args.args[0].equals(pd.DataFrame({'raw': [1]}))
        assert mock_odds.call_args.args[1] == feature_pipeline.season
        assert mock_odds.call_args.args[2] is True
        assert valid_schedule_seen_by_match['game_id'].tolist() == ['game1', 'game2']

        expected_index = pd.MultiIndex.from_frame(pd.DataFrame(matched_base_rows)[idx])
        assert final_features.index.equals(expected_index)
        assert len(final_features) == 2
        assert final_features.index.get_level_values('game_date').is_monotonic_increasing

        matched_schedule_arg = mock_get_position.call_args.args[0]
        assert matched_schedule_arg.index.names == idx
        assert matched_schedule_arg.index.equals(expected_index)
        assert mock_get_position.call_args.args[1] is True

        pitcher_schedule_arg = mock_get_pitcher.call_args.args[0]
        assert pitcher_schedule_arg.index.equals(expected_index)
        assert mock_get_pitcher.call_args.args[1] is True

        context_schedule_arg = mock_context_cls.call_args.args[1]
        assert context_schedule_arg['game_id'].tolist() == ['game1', 'game2']

        team_schedule_arg = mock_team_cls.call_args.args[1]
        assert team_schedule_arg.index.equals(expected_index)

        assert 'wind' not in final_features.columns
        assert 'condition' not in final_features.columns
        assert 'park_factor' in final_features.columns
        assert 'vig_open' in final_features.columns

        first = final_features.loc[('game1', pd.Timestamp('2024-04-01'), 0, 'BOS', 'NYY')]
        assert first['home_away_starter_era_season_diff'] == pytest.approx(
            first['home_starter_era_season'] - first['away_starter_era_season']
        )
        assert first['home_away_pen_fip_ewm_h8_diff'] == pytest.approx(
            first['home_pen_fip_ewm_h8'] - first['away_pen_fip_ewm_h8']
        )
        assert first['home_away_starter_fip_woba_season_diff'] == pytest.approx(
            first['home_starter_fip_season'] - first['away_woba_season']
        )
        assert first['home_away_starter_k_percent_k_percent_ewm_h3_diff'] == pytest.approx(
            first['home_starter_k_percent_ewm_h3'] - first['away_k_percent_ewm_h3']
        )
        assert first['home_away_win_pct_ewm_h20_diff'] == pytest.approx(
            first['home_win_pct_ewm_h20'] - first['away_win_pct_ewm_h20']
        )
        assert first['home_away_one_run_win_pct_season_diff'] == pytest.approx(
            first['home_one_run_win_pct_season'] - first['away_one_run_win_pct_season']
        )

        assert not final_features.isna().any().any()
        mock_get_position.assert_called_once()

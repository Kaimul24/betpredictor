"""
Tests for GameContextFeatures class.

Tests the game context features including:
- Weather feature encoding (condition, wind, temperature)
- Venue and park factor features
- Day/night game encoding
- Proper one-hot encoding for categorical variables
- Handling of missing/null values
- Feature engineering pipeline integration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime

from data.features.game_features.context import GameContextFeatures
from tests.conftest import (
    insert_schedule_games, insert_park_factors, assert_dataframe_schema,
    assert_dataframe_not_empty
)


class TestGameContextFeatures:
    """Test suite for GameContextFeatures class."""

    @pytest.fixture
    def sample_schedule_data(self, clean_db):
        """Sample schedule data with various weather conditions and venues."""
        games = [
            ('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS', 
             'Final', 5, 3, 'NYY', 'BOS', 0),
            
            ('game2', '2024-04-02', '2024-04-02T13:10:00', 2024, 'TB', 'NYY', 
             'Final', 2, 7, 'NYY', 'TB', 0),
            
            ('game3', '2024-04-03', '2024-04-03T19:35:00', 2024, 'LAD', 'SF', 
             'Final', 4, 1, 'LAD', 'SF', 0),
        ]
        insert_schedule_games(clean_db, games)
        
        from data.database import get_database_manager
        db = get_database_manager()
        
        updates = [
            ("day", 1, "Yankee Stadium", 33, 78, "Sunny", "15 mph, Out To RF", "game1"),
            ("night", 2, "Fenway Park", 21, 65, "Overcast", "5 mph, Calm", "game2"),
            ("day", 3, "Coors Field", 5280, 72, None, None, "game3"),
        ]
        
        for day_night, venue_id, venue_name, elevation, temp, condition, wind, game_id in updates:
            update_query = """
            UPDATE schedule 
            SET day_night_game = ?, venue_id = ?, venue_name = ?, 
                venue_elevation = ?, temp = ?, condition = ?, wind = ?
            WHERE game_id = ?
            """
            db.execute_write_query(update_query, (day_night, venue_id, venue_name, elevation, temp, condition, wind, game_id))
        
        return games

    @pytest.fixture
    def sample_park_factors(self, clean_db):
        """Sample park factor data."""
        park_factors = [
            (1, "Yankee Stadium", 2024, 105),
            (2, "Fenway Park", 2024, 98), 
            (3, "Coors Field", 2024, 115),
        ]
        insert_park_factors(clean_db, park_factors)
        return park_factors

    @pytest.fixture
    def context_features(self, clean_db, sample_schedule_data, sample_park_factors):
        """Create GameContextFeatures instance with sample data."""
        from data.loaders.game_loader import GameLoader
        loader = GameLoader()
        schedule_data = loader.load_for_season(2024)
        
        return GameContextFeatures(data=schedule_data, season=2024)

    def test_init(self, context_features):
        """Test GameContextFeatures initialization."""
        assert context_features.season == 2024
        assert hasattr(context_features, 'park_factor_data')
        assert not context_features.park_factor_data.empty

    def test_load_data(self, context_features):
        """Test load_data returns the original data."""
        original_data = context_features.data.copy()
        loaded_data = context_features.load_data()
        
        pd.testing.assert_frame_equal(loaded_data, original_data)

    def test_day_night_encoding(self, context_features):
        """Test binary encoding of day/night games."""
        day_night_encoded = context_features._day_night_game()
        
        assert_dataframe_not_empty(day_night_encoded)
        assert 'day_night_game' in day_night_encoded.columns
        
        unique_values = day_night_encoded['day_night_game'].unique()
        assert all(val in [0, 1] for val in unique_values), \
            f"Day/night encoding should be 0 or 1, got {unique_values}"
        
        original_data = context_features.data.reset_index()
        for i, row in day_night_encoded.iterrows():
            original_value = original_data.iloc[i]['day_night_game']
            encoded_value = row['day_night_game']
            
            if original_value == 'day':
                assert encoded_value == 1, "Day games should be encoded as 1"
            elif original_value == 'night':
                assert encoded_value == 0, "Night games should be encoded as 0"

    def test_wind_encoding(self, context_features):
        """Test wind feature encoding including magnitude and direction."""
        wind_encoded = context_features._encode_wind()
        
        assert_dataframe_not_empty(wind_encoded)
        
        assert 'wind_magnitude' in wind_encoded.columns
        assert wind_encoded['wind_magnitude'].dtype.name.startswith('int')
        
        expected_directions = [
            "wind_direction_None", "wind_direction_Calm", "wind_direction_In_From_LF",
            "wind_direction_In_From_CF", "wind_direction_In_From_RF", "wind_direction_Out_To_LF",
            "wind_direction_Out_To_RF", "wind_direction_Out_To_CF", "wind_direction_R_To_L",
            "wind_direction_L_To_R", "wind_direction_Varies"
        ]
        
        for direction_col in expected_directions:
            assert direction_col in wind_encoded.columns, f"Missing wind direction column: {direction_col}"
            assert wind_encoded[direction_col].dtype == 'int64', \
                f"Wind direction {direction_col} should be int64"
            assert all(val in [0, 1] for val in wind_encoded[direction_col].unique()), \
                f"Wind direction {direction_col} should be one-hot encoded (0 or 1)"

    def test_wind_magnitude_parsing(self, context_features):
        """Test that wind magnitude is correctly parsed from wind strings."""
        wind_encoded = context_features._encode_wind()
        
        original_data = context_features.data.reset_index()
        
        for i, row in wind_encoded.iterrows():
            original_wind = original_data.iloc[i]['wind']
            magnitude = row['wind_magnitude']
            
            if pd.notna(original_wind) and original_wind:
                wind_parts = original_wind.split(',')[0].strip().split()

                if wind_parts and wind_parts[0].replace('mph', '').isdigit():
                    expected_magnitude = int(wind_parts[0].replace('mph', ''))
                    assert magnitude == expected_magnitude, \
                        f"Wind magnitude should be {expected_magnitude}, got {magnitude}"
            else:
                assert magnitude == 0, "Missing wind should default to 0 magnitude"

    def test_condition_encoding(self, context_features):
        """Test weather condition one-hot encoding."""
        condition_encoded = context_features._encode_condition()
        
        assert_dataframe_not_empty(condition_encoded)
        
        expected_conditions = [
            "condition_Cloudy", "condition_Snow", "condition_Roof_Closed",
            "condition_Sunny", "condition_Partly_Cloudy", "condition_Clear",
            "condition_Rain", "condition_Overcast", "condition_Dome",
            "condition_Drizzle", "condition_Unknown"
        ]
        
        for condition_col in expected_conditions:
            assert condition_col in condition_encoded.columns, \
                f"Missing condition column: {condition_col}"
            assert condition_encoded[condition_col].dtype == 'int64', \
                f"Condition {condition_col} should be int64"
            assert all(val in [0, 1] for val in condition_encoded[condition_col].unique()), \
                f"Condition {condition_col} should be one-hot encoded (0 or 1)"

    def test_condition_unknown_handling(self, context_features):
        """Test that missing conditions are handled as 'Unknown'."""
        condition_encoded = context_features._encode_condition()
        original_data = context_features.data.reset_index()
        
        for i, row in condition_encoded.iterrows():
            original_condition = original_data.iloc[i]['condition']
            
            if pd.isna(original_condition):
                assert row['condition_Unknown'] == 1, \
                    "Missing condition should be encoded as Unknown"
                
                other_conditions = [col for col in condition_encoded.columns 
                                  if col.startswith('condition_') and col != 'condition_Unknown']
                assert all(row[col] == 0 for col in other_conditions), \
                    "Only Unknown condition should be 1 for missing values"

    def test_weather_features_integration(self, context_features):
        """Test that weather features are properly integrated."""
        weather_features = context_features._create_weather_features()
        
        assert_dataframe_not_empty(weather_features)
        assert 'temp' in weather_features.columns
        
        wind_cols = [col for col in weather_features.columns if col.startswith('wind_')]
        assert len(wind_cols) > 0, "Should contain wind features"
        
        condition_cols = [col for col in weather_features.columns if col.startswith('condition_')]
        assert len(condition_cols) > 0, "Should contain condition features"

    def test_park_factor_mapping(self, context_features):
        """Test that park factors are correctly mapped to venues."""
        features = context_features.load_features()
        
        assert 'park_factor' in features.columns
        
        venue_to_factor = {1: 105, 2: 98, 3: 115}
        
        for _, row in features.iterrows():
            venue_id = row['venue_id']
            park_factor = row['park_factor']
            
            if venue_id in venue_to_factor:
                expected_factor = venue_to_factor[venue_id]
                assert park_factor == expected_factor, \
                    f"Venue {venue_id} should have park factor {expected_factor}, got {park_factor}"

    def test_load_features_complete_pipeline(self, context_features):
        """Test the complete feature loading pipeline."""
        features = context_features.load_features()
        
        assert_dataframe_not_empty(features)
        
        essential_cols = ['game_id', 'game_date', 'game_datetime', 'away_team', 
                         'home_team', 'dh', 'park_factor', 'venue_elevation']
        assert_dataframe_schema(features, essential_cols)
        
        assert 'temp' in features.columns
        assert any(col.startswith('wind_') for col in features.columns)
        assert any(col.startswith('condition_') for col in features.columns)
        
        assert 'day_night_game' in features.columns
        assert all(val in [0, 1] for val in features['day_night_game'].unique())

    def test_missing_condition_column_error(self, clean_db):
        """Test that missing condition column raises appropriate error."""
        games = [('game1', '2024-04-01', '2024-04-01T19:05:00', 2024, 'NYY', 'BOS', 
                 'Final', 5, 3, 'NYY', 'BOS', 0)]
        insert_schedule_games(clean_db, games)
        
        from data.loaders.game_loader import GameLoader
        loader = GameLoader()
        schedule_data = loader.load_for_season(2024)
        
        if 'condition' in schedule_data.columns:
            schedule_data = schedule_data.drop(columns=['condition'])
        
        context_features = GameContextFeatures(data=schedule_data, season=2024)
        
        with pytest.raises(ValueError, match="'condition' column not found in data"):
            context_features._encode_condition()

    def test_elevation_feature(self, context_features):
        """Test that venue elevation is included in features."""
        features = context_features.load_features()
        
        assert 'venue_elevation' in features.columns
        
        expected_elevations = {1: 33, 2: 21, 3: 5280}
        
        for _, row in features.iterrows():
            venue_id = row['venue_id']
            elevation = row['venue_elevation']
            
            if venue_id in expected_elevations:
                expected_elevation = expected_elevations[venue_id]
                assert elevation == expected_elevation, \
                    f"Venue {venue_id} should have elevation {expected_elevation}, got {elevation}"

    def test_feature_dtypes(self, context_features):
        """Test that all features have appropriate data types."""
        features = context_features.load_features()
        
        categorical_cols = [col for col in features.columns 
                          if col.startswith(('wind_direction_', 'condition_', 'day_night_game'))]
        
        for col in categorical_cols:
            if col in features.columns:
                assert features[col].dtype == 'int64', \
                    f"Categorical feature {col} should be int64, got {features[col].dtype}"
        
        if 'wind_magnitude' in features.columns:
            assert features['wind_magnitude'].dtype.name.startswith('int'), \
                f"Wind magnitude should be int, got {features['wind_magnitude'].dtype}"

    def test_no_data_leakage_in_features(self, context_features):
        """Test that context features don't contain future information."""
        features = context_features.load_features()
        
        leakage_cols = ['away_score', 'home_score', 'winning_team', 'losing_team', 'winner']
        
        for col in leakage_cols:
            assert col not in features.columns, \
                f"Context features should not contain outcome information: {col}"

    def test_one_hot_encoding_mutually_exclusive(self, context_features):
        """Test that one-hot encoded features are mutually exclusive."""
        wind_encoded = context_features._encode_wind()
        condition_encoded = context_features._encode_condition()
        
        wind_direction_cols = [col for col in wind_encoded.columns if col.startswith('wind_direction_')]
        wind_sums = wind_encoded[wind_direction_cols].sum(axis=1)
        assert all(wind_sums == 1), "Wind direction encoding should be mutually exclusive"
        
        condition_cols = [col for col in condition_encoded.columns if col.startswith('condition_')]
        condition_sums = condition_encoded[condition_cols].sum(axis=1)
        assert all(condition_sums == 1), "Condition encoding should be mutually exclusive"

"""
Handles weather, venue/park factors, game time metrics.
"""

import pandas as pd
import logging
from pandas.core.api import DataFrame as DataFrame
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta

from data.features.base_feature import BaseFeatures
from data.loaders.game_loader import GameLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameContextFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame) -> None:
        super().__init__(season, data)

        self.park_factor_data = GameLoader().load_park_factor_season(season)

    def load_data(self) -> DataFrame:
        return self.data
    
    def load_features(self) -> DataFrame:
        """
        Performs feature engineering for game context features such as weather,
        venue, park factors, day/night game, etc. 
        """
        self.data = self.data.reset_index().copy()
        weather_features = self._create_weather_features()
        day_night_col = self._day_night_game()

        venue_to_park_factor = self.park_factor_data.set_index('venue_id')['park_factor'].copy()
        self.data['park_factor'] = self.data['venue_id'].map(venue_to_park_factor)

        essential_cols = ['game_id', 'game_date', 'game_datetime', 'away_team', 'home_team', 'dh', 'venue_name', 'venue_id', 'park_factor', 'venue_elevation']
        context_data = self.data[essential_cols].copy()

        return pd.concat([context_data.reset_index(drop=True),
            weather_features.reset_index(drop=True),
            day_night_col.reset_index(drop=True)], axis=1
        )
    
    def _day_night_game(self) -> DataFrame:
        """Binary encoding of the day_night column. 1 is day, 0 is night."""
        day_night_col = self.data[['day_night_game']].copy()
        day_night_game = (day_night_col == "day").astype(int)
        return day_night_game
    
    def _create_weather_features(self) -> DataFrame:
        "Returns all weather features. Calls helper functions for condition and wind"
        wind_cols = self._encode_wind()
        condition_cols = self._encode_condition()
        temp = self.data['temp']
        return pd.concat([temp.reset_index(drop=True),
                            condition_cols.reset_index(drop=True),
                            wind_cols.reset_index(drop=True)], axis=1)

    def _encode_wind(self) -> DataFrame:
        """Encodes the wind column."""
        wind_encoded = self.data[['wind']].copy()
        wind_encoded[['wind_magnitude' , 'wind_direction']] \
            = wind_encoded['wind'].str.split(',', expand=True)
        
        wind_encoded['wind_magnitude'] = wind_encoded['wind_magnitude'].str.split().str[0]
        wind_encoded['wind_magnitude'] = wind_encoded['wind_magnitude'].fillna("0")
        
        wind_encoded['wind_magnitude'] = pd.to_numeric(wind_encoded['wind_magnitude'], errors='coerce').fillna(0).astype(int)
        
        wind_encoded['wind_direction'] = wind_encoded['wind_direction'].fillna("None")
        wind_encoded['wind_direction'] = wind_encoded['wind_direction'].str.strip()

        categories = [
            "None", 
            "Calm", 
            "In From LF", 
            "In From CF", 
            "In From RF",
            "Out To LF",
            "Out To RF",
            "Out To CF",
            "R To L", 
            "L To R", 
            "Varies"
        ]

        encoder = OneHotEncoder(
            categories=[categories],
            handle_unknown='ignore',
            sparse_output=False,
            dtype=int,
        )

        encoded = encoder.fit_transform(wind_encoded[['wind_direction']])
        col_names = [f"wind_direction_{cat.replace(' ', '_')}" for cat in categories]
        encoded_df = DataFrame(encoded, columns=col_names, index=self.data.index)
        
        encoded_df['wind_magnitude'] = wind_encoded['wind_magnitude']

        assert encoded_df['wind_magnitude'].dtype.name.startswith('int'), f"Expected int dtype, got {encoded_df['wind_magnitude'].dtype}"
        assert all(encoded_df[f"wind_direction_{cat.replace(' ', '_')}"].dtype == 'int64' for cat in categories), "All wind direction columns should be int64"
        
        return encoded_df

    def _encode_condition(self) -> DataFrame:
        """One-hot encode the 'condition' column."""

        if 'condition' not in self.data.columns:
            raise ValueError("'condition' column not found in data")

        categories = [
            "Cloudy",
            "Snow",
            "Roof Closed",
            "Sunny",
            "Partly Cloudy",
            "Clear",
            "Rain",
            "Overcast",
            "Dome",
            "Drizzle",
            "Unknown",
        ]

        X = self.data[['condition']].copy()
        X['condition'] = X['condition'].fillna('Unknown')

        encoder = OneHotEncoder(
            categories=[categories],
            handle_unknown='ignore',
            sparse_output=False,
            dtype=int,
        )

        encoded = encoder.fit_transform(X)
        col_names = [f"condition_{cat.replace(' ', '_')}" for cat in categories]
        encoded_condition = pd.DataFrame(encoded, columns=col_names, index=self.data.index)
        assert all(encoded_condition[f"condition_{cat.replace(' ', '_')}"].dtype == 'int64' for cat in categories), "All wind direction columns should be int64"
        return encoded_condition

def main():
    from data.loaders.game_loader import GameLoader
    game_loader = GameLoader()
    game_data = game_loader.load_for_season(2021)
    context_features = GameContextFeatures(game_data, 2021).load_features()
    print(context_features.columns)
if __name__ == "__main__":
    main()    






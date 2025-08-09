"""
Handles weather, venue/park factors, game time metrics.
"""

import pandas as pd
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from sklearn.preprocessing import OneHotEncoder

from data.database import get_database_manager


class GameContextFeatures(BaseFeatures):

    def __init__(self, season: int, schedule_data: DataFrame) -> None:
        super().__init__(season)
        self.schedule_data = schedule_data

    def load_data(self) -> DataFrame:
        return self.schedule_data
    
    def weather_features(self) -> DataFrame:
        pass

    def _get_park_factor(self):
        pass

    def _relative_game_time(self):
        pass

    

    def _encode_wind(self) -> DataFrame:
        """Encodes the wind column."""
        wind_encoded = self.schedule_data[['wind']].copy()
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
        encoded_df = DataFrame(encoded, columns=col_names, index=self.schedule_data.index)
        
        encoded_df['wind_magnitude'] = wind_encoded['wind_magnitude']

        assert encoded_df['wind_magnitude'].dtype.name.startswith('int'), f"Expected int dtype, got {encoded_df['wind_magnitude'].dtype}"
        assert all(encoded_df[f"wind_direction_{cat.replace(' ', '_')}"].dtype == 'int64' for cat in categories), "All wind direction columns should be int64"
        
        return encoded_df

    def _encode_condition(self) -> DataFrame:
        """One-hot encode the 'condition' column."""

        if 'condition' not in self.schedule_data.columns:
            raise ValueError("'condition' column not found in schedule_data")

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

        X = self.schedule_data[['condition']].copy()
        X['condition'] = X['condition'].fillna('Unknown')

        encoder = OneHotEncoder(
            categories=[categories],
            handle_unknown='ignore',
            sparse_output=False,
            dtype=int,
        )

        encoded = encoder.fit_transform(X)
        col_names = [f"condition_{cat.replace(' ', '_')}" for cat in categories]
        encoded_condition = pd.DataFrame(encoded, columns=col_names, index=self.schedule_data.index)
        assert all(encoded_condition[f"condition_{cat.replace(' ', '_')}"].dtype == 'int64' for cat in categories), "All wind direction columns should be int64"
        return encoded_condition
       

if __name__ == "__main__":
    db_manager = get_database_manager()
    # cols = ['wind', 'condition', 'temp']
    # for col in cols:
    #     query = f"""
    #     SELECT DISTINCT {col}
    #     FROM schedule;
    #     """
    #     result = db_manager.execute_read_query(query)
    #     df = DataFrame([dict(row) for row in result])
    #     print(df)
    query = """
    SELECT * FROM schedule;
    """
    result = db_manager.execute_read_query(query)
    schedule_data = DataFrame([dict(row) for row in result])

    context_feats = GameContextFeatures(2021, schedule_data)
    condition_data = context_feats._encode_condition()
    print(condition_data)

    wind_data = context_feats._encode_wind()
    print(wind_data)






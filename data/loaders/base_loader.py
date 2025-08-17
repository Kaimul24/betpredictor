from abc import ABC, abstractmethod
from datetime import date, datetime
from pandas.core.api import DataFrame as DataFrame
import pandas as pd
import logging
from data.database import get_database_manager


class BaseDataLoader(ABC):
    def __init__(self):
        self.db_manager = get_database_manager()
    
    @abstractmethod
    def load_for_date_range(self, start: date, end: date) -> DataFrame:
        pass

    @abstractmethod
    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        pass

    @abstractmethod
    def load_for_season(self, season: int) -> DataFrame:
        pass

    def _time_filter(self, date: date, dh: int = 0) -> tuple[str, list]:
        """Generate time-based filter conditions for doubleheader handling."""
        if dh <= 1:
            return "game_date < ?", [date.strftime('%Y-%m-%d')]
        else:
            return "game_date < ? OR (game_date = ? AND dh < ?)", [
                date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), dh
            ]

    def _time_filter_season(self, date: date, dh: int = 0) -> tuple[str, list]:
        """Filter for current season only up to specified date."""
        year = date.year
        if dh <= 1:
            return "year = ? AND game_date < ?", [year, date.strftime('%Y-%m-%d')]
        else:
            return "year = ? AND (game_date < ? OR (game_date = ? AND dh < ?))", [
                year, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), dh
            ]

        
    def _execute_query(self, query: str, params: list | None = None) -> DataFrame:
        """Execute query using database manager."""
        try:
            results = self.db_manager.execute_read_query(query, tuple(params or []))
            if results:
                df = DataFrame([dict(row) for row in results])
                return df
            else:
                return DataFrame()
        except Exception as e:
            logging.error(f"Query failed: {e}\nQuery: {query}\nParams: {params}")
            return DataFrame()

    def _validate_dataframe(self, df: DataFrame, required_columns: list[str]) -> DataFrame:
        """
        Validate loaded data meets requirements.
        
        Checks:
        1. Required columns exist
        2. No completely empty columns
        3. Date columns are properly formatted
        4. Numerical columns have valid values (no inf, etc.)
        """
        if df.empty:
            return df
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        for col in df.columns:
            if 'game_date' in col.lower() or 'game_datetime' in col.lower():

                df[col] = pd.to_datetime(df[col]).astype('datetime64[ns]')
            elif df[col].dtype in ['float64', 'int64']:
                
                if df[col].isin([float('inf'), float('-inf')]).any():
                    print(f"Infinite values found in {col}")
                    df[col] = df[col].replace([float('inf'), float('-inf')], None)
                    
        return df
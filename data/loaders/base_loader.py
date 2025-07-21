from abc import ABC, abstractmethod
from datetime import date, datetime
import pandas as pd
import logging
from data.database import get_database_manager


class BaseDataLoader(ABC):
    def __init__(self):
        self.db_manager = get_database_manager()
    
    @abstractmethod
    def load_for_date_range(self, start: date, end: date) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_up_to_game(self, date: date, team_id: str, dh: int = 0) -> pd.DataFrame:
        pass

    def _time_filter(self, date: date, dh: int = 0) -> tuple[str, list]:
        """Generate time-based filter conditions for doubleheader handling."""
        if dh <= 1:
            return "game_date < ?", [date.strftime('%Y-%m-%d')]
        else:
            return "game_date < ? OR (game_date = ? AND game_num < ?)", [
                date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), dh
            ]

    def _time_filter_season(self, date: date, dh: int = 0) -> tuple[str, list]:
        """Filter for current season only up to specified date."""
        year = date.year
        if dh <= 1:
            return "year = ? AND game_date < ?", [year, date.strftime('%Y-%m-%d')]
        else:
            return "year = ? AND (game_date < ? OR (game_date = ? AND game_num < ?))", [
                year, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), dh
            ]

        
    def _execute_query(self, query: str, params: list = None) -> pd.DataFrame:
        """Execute query using database manager."""
        try:
            results = self.db_manager.execute_read_query(query, tuple(params or []))
            if results:
                df = pd.DataFrame([dict(row) for row in results])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Query failed: {e}\nQuery: {query}\nParams: {params}")
            return pd.DataFrame()

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
        """
        Validate loaded data meets requirements.
        
        Checks:
        1. Required columns exist
        2. No completely empty columns
        3. Date columns are properly formatted
        4. Numerical columns have valid values (no inf, etc.)
        """
        # Check required columns
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Validate data types and values
        for col in df.columns:
            if 'date' in col.lower():
                # Ensure dates are parsed correctly
                df[col] = pd.to_datetime(df[col])
            elif df[col].dtype in ['float64', 'int64']:
                # Check for infinite values
                if df[col].isin([float('inf'), float('-inf')]).any():
                    print(f"Infinite values found in {col}")
                    df[col] = df[col].replace([float('inf'), float('-inf')], None)
                    
        return df
'''
Game and schedule data loader
'''

from src.data.loaders.base_loader import BaseDataLoader
from datetime import date
from typing import Union
from itertools import islice
from pandas.core.api import DataFrame as DataFrame
import pandas as pd

class GameLoader(BaseDataLoader):

    def __init__(self):
        super().__init__()
        self.columns = ['game_id', 'game_date', 'game_datetime', 'day_night_game', 'season','away_team', 
                        'home_team', 'dh', 'venue_name', 'venue_id', 'venue_elevation', 'venue_timezone',
                        'venue_gametime_offset', 'status', 'away_probable_pitcher','home_probable_pitcher', 
                        'away_starter_normalized', 'home_starter_normalized', 'away_pitcher_id', 
                        'home_pitcher_id', 'wind', 'condition', 'temp',  'away_score', 'home_score', 
                        'winning_team', 'losing_team']
        
        self.venue_columns = ['venue_id', 'venue_name', 'season', 'park_factor']
        
    def load_for_season(self, season: int) -> DataFrame:
        """Load all games for a season"""
        params = [season]
        columns_str = ",\n\t".join(self.columns)
        query = f"""
        SELECT 
        \t{columns_str}
        FROM schedule
        WHERE season = ?
        ORDER BY game_date, dh
        """
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.columns)


    def load_for_date_range(self, start: date, end: date) -> DataFrame:
        """
        Load all games in a range of dates
        """
        columns_str = ",\n\t".join(self.columns)
        query = f"""
        SELECT 
        \t{columns_str}
        FROM schedule
        WHERE game_date BETWEEN ? and ?
        ORDER BY game_date, dh
        """
        df = self._execute_query(query, [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')])
        return self._validate_dataframe(df, self.columns)

    
    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        """
        Load all games for a team up until date. Handles doubleheaders.
        """
        where, params = self._time_filter(date, dh)
        columns_str = ",\n\t".join(self.columns)

        query = f"""
        SELECT 
        \t{columns_str},
            CASE 
                WHEN home_team = ? THEN 'home'
                WHEN away_team = ? THEN 'away'
            END as team_side
        FROM schedule
        WHERE ({where})
            AND (home_team = ? OR away_team = ?)
        ORDER BY game_date, dh
        """
        
        all_params = [team_abbr, team_abbr] + params + [team_abbr, team_abbr]
        df = self._execute_query(query, all_params)
        return self._validate_dataframe(df, self.columns)
    
    def load_season_games(self, season: int, team_abbr: str = None) -> DataFrame:
        """
        Load all games for a specific season, optionally filtered by team.
        """
        columns_str = ",\n\t".join(self.columns)
        if team_abbr:
            query = f"""
            SELECT 
            \t{columns_str}
            FROM schedule 
            WHERE season = ? AND (home_team = ? OR away_team = ?)
            ORDER BY game_date, dh
            """
            params = [season, team_abbr, team_abbr]
        else:
            query = f"""
            SELECT \t{columns_str} 
            FROM schedule 
            WHERE season = ?
            ORDER BY game_date, dh
            """
            params = [season]
        
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.columns)

    def load_up_to_game_season(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        """
        Load all games for a team up until date within the same season only.
        """
        season = date.year
        where, params = self._time_filter(date, dh)
        columns_str = ",\n\t".join(self.columns)
        
        query = f"""
        SELECT 
        \t{columns_str},
            CASE 
                WHEN home_team = ? THEN 'home'
                WHEN away_team = ? THEN 'away'
            END as team_side,
            CASE
                WHEN winning_team = ? THEN 1
                ELSE 0
            END as team_won
        FROM schedule
        WHERE season = ? AND ({where})
            AND (home_team = ? OR away_team = ?)
        ORDER BY game_date, dh
        """
        
        all_params = [team_abbr] * 3 + [season] + params + [team_abbr] * 2
        df = self._execute_query(query, all_params)
        return self._validate_dataframe(df, self.columns)
    
    def load_park_factor_season(self, season: int) -> DataFrame:
        """
        Load park factor metrics for a season
        """
        columns_str = ",\n\t".join(self.venue_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM park_factors
        WHERE season = ? 
            AND park_factor IS NOT NULL;
        """
        df = self._execute_query(query, params=[season])
        return self._validate_dataframe(df, self.venue_columns)


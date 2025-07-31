'''
Game and schedule data loader
'''

from data.loaders.base_loader import BaseDataLoader
from datetime import date
from typing import Union
from itertools import islice
import pandas as pd

class GameLoader(BaseDataLoader):

    def __init__(self):
        super().__init__()
        self.columns = ['game_id', 'game_date', 'game_datetime', 'season', 'away_team', 'home_team',
                    'away_team_abbr', 'home_team_abbr', 'doubleheader', 'game_num', 'venue_name', 
                    'venue_id', 'status', 'away_probable_pitcher', 'home_probable_pitcher', 'wind', 
                    'condition', 'temp', 'away_score', 'home_score', 'winning_team', 'losing_team']
        

    def load_for_date_range(self, start: date, end: date) -> pd.DataFrame:
        """
        Load all games in a range of dates
        """
        query = """
        SELECT 
            game_id,
            game_date, 
            game_datetime,
            season,
            away_team, 
            home_team, 
            away_team_abbr, 
            home_team_abbr, 
            doubleheader, 
            game_num, 
            venue_name, 
            venue_id, 
            status, 
            away_probable_pitcher, 
            home_probable_pitcher, 
            wind, 
            condition, 
            temp,
            away_score,
            home_score,
            winning_team,
            losing_team
        FROM schedule
        WHERE game_date BETWEEN ? and ?
        ORDER BY game_date, game_num
        """
        df = self._execute_query(query, [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')])
        return self._validate_dataframe(df, self.columns)

    
    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> pd.DataFrame:
        """
        Load all games for a team up until date. Handles doubleheaders.
        """
        where, params = self._time_filter(date, dh)

        query = f"""
        SELECT 
            *,
            CASE 
                WHEN home_team_abbr = ? THEN 'home'
                WHEN away_team_abbr = ? THEN 'away'
            END as team_side
        FROM schedule
        WHERE ({where})
            AND (home_team_abbr = ? OR away_team_abbr = ?)
        ORDER BY game_date, game_num
        """
        
        all_params = [team_abbr, team_abbr] + params + [team_abbr, team_abbr]
        return self._execute_query(query, all_params)
    
    def load_season_games(self, season: int, team_abbr: str = None) -> pd.DataFrame:
        """
        Load all games for a specific season, optionally filtered by team.
        """
        if team_abbr:
            query = """
            SELECT * FROM schedule 
            WHERE season = ? AND (home_team_abbr = ? OR away_team_abbr = ?)
            ORDER BY game_date, game_num
            """
            params = [season, team_abbr, team_abbr]
        else:
            query = """
            SELECT * FROM schedule 
            WHERE season = ?
            ORDER BY game_date, game_num
            """
            params = [season]
        
        return self._execute_query(query, params)

    def load_up_to_game_season(self, date: date, team_abbr: str, dh: int = 0) -> pd.DataFrame:
        """
        Load all games for a team up until date within the same season only.
        """
        season = date.year
        where, params = self._time_filter(date, dh)
        
        query = f"""
        SELECT 
            *,
            CASE 
                WHEN home_team_abbr = ? THEN 'home'
                WHEN away_team_abbr = ? THEN 'away'
            END as team_side,
            CASE
                WHEN winning_team = ? THEN 1
                ELSE 0
            END as team_won
        FROM schedule
        WHERE season = ? AND ({where})
            AND (home_team_abbr = ? OR away_team_abbr = ?)
        ORDER BY game_date, game_num
        """
        
        all_params = [team_abbr] * 3 + [season] + params + [team_abbr] * 2
        return self._execute_query(query, all_params)


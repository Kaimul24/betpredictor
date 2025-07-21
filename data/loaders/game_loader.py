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
                    'away_team_abbr', 'home_team_abbr', 'doubleheader', 'game_num', 
                    'venue_name', 'venue_id', 'status', 'away_probable_pitcher', 
                    'home_probable_pitcher', 'wind', 'condition', 'temp', 'away_score', 'home_score']
        

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
            home_score
        FROM schedule
        WHERE game_date BETWEEN ? and ?
            AND status = 'Final'
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
                WHEN home_team = ? THEN 'home'
                WHEN away_team = ? THEN 'away'
            END as team_side,
            CASE
                WHEN (home_team = ? AND home_score > away_score) OR 
                     (away_team = ? AND away_score > home_score) THEN 1
                ELSE 0
            END as team_won
        FROM schedule
        WHERE ({where})
            AND (home_team = ? OR away_team = ?)
            AND status = 'Final'
        ORDER BY game_date, game_num
        """
        
        all_params = params + [team_abbr] * 6
        return self._execute_query(query, all_params)
    
    def load_season_games(self, season: int, team_abbr: str = None) -> pd.DataFrame:
        """
        Load all games for a specific season, optionally filtered by team.
        """
        if team_abbr:
            query = """
            SELECT * FROM schedule 
            WHERE season = ? AND (home_team_abbr = ? OR away_team_abbr = ?)
                AND status = 'Final'
            ORDER BY game_date, game_num
            """
            params = [season, team_abbr, team_abbr]
        else:
            query = """
            SELECT * FROM schedule 
            WHERE season = ? AND status = 'Final'
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
                WHEN (home_team_abbr = ? AND home_score > away_score) OR 
                     (away_team_abbr = ? AND away_score > home_score) THEN 1
                ELSE 0
            END as team_won
        FROM schedule
        WHERE season = ? AND ({where})
            AND (home_team_abbr = ? OR away_team_abbr = ?)
            AND (status = 'Final' OR status = 'Completed Early: Rain')
        ORDER BY game_date, game_num
        """
        
        all_params = [team_abbr] * 4 + [season] + params + [team_abbr] * 2
        return self._execute_query(query, all_params)

    def team_record(self, date: date, team_abbr: str) -> dict[str, Union[int, float]]:
        """
        Get team's win-loss record up to a specific date within the season.
        """
        df = self.load_up_to_game_season(date, team_abbr)
        if df.empty:
            return {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'games': 0}
        
        wins = df['team_won'].sum()
        losses = len(df) - wins
        win_pct = wins / len(df) if len(df) > 0 else 0.0
        
        return {
            'wins': int(wins),
            'losses': int(losses), 
            'games': len(df),
            'win_pct': float(win_pct)
        }
        
    def game_streak(self, date: date, team_abbr: str, num_games: int = 10) -> int:
        """
        Get current win/loss streak for a team up to a specific date.
        Returns positive for win streak, negative for loss streak.
        """
        df = self.load_up_to_game_season(date, team_abbr)
        if df.empty:
            return 0
        
        df = df.sort_values(['game_date', 'game_num'], ascending=False)
        
        streak = 0
        last_result = None
        
        for _, game in islice(df.iterrows(), num_games):
            won = bool(game['team_won'])

            if last_result == None or last_result != won:
                    last_result = won
                    streak = 1 if won else -1
            else:
                    last_result = won
                    streak = streak + 1 if won else streak -1
                    
            return streak

    def rest_days(self, date: date, team_abbr: str) -> int:
        """
        Calculate rest days between the target date and team's last game.
        """
        query = """
        SELECT MAX(game_date) as last_game_date
        FROM schedule
        WHERE (home_team = ? OR away_team = ?)
            AND game_date < ?
            AND status = 'Final'
        """
        
        df = self._execute_query(query, [team_abbr, team_abbr, date.strftime('%Y-%m-%d')])
        
        if df.empty or pd.isna(df.iloc[0]['last_game_date']):
            return 0
            
        last_game = pd.to_datetime(df.iloc[0]['last_game_date']).date()
        return (date - last_game).days - 1
    



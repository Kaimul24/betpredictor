import pandas as pd
from typing import Union
from datetime import date
from .base_loader import BaseDataLoader
from .game_loader import GameLoader


class TeamLoader(BaseDataLoader):

    def __init__(self):
        super().__init__()
        
        self.game_loader = GameLoader()

    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> pd.DataFrame:
        return super().load_up_to_game(date, team_abbr, dh)
    
    def load_for_date_range(self, start: date, end: date) -> pd.DataFrame:
        return super().load_for_date_range(start, end)
    
    def rest_days(self, date: date, team_abbr: str) -> int:
        """
        Calculate rest days between the target date and team's last game.
        """
        query = """
        SELECT MAX(game_date) as last_game_date
        FROM schedule
        WHERE (home_team = ? OR away_team = ?)
            AND game_date < ?
        """
        
        df = self._execute_query(query, [team_abbr, team_abbr, date.strftime('%Y-%m-%d')])
        
        if df.empty or pd.isna(df.iloc[0]['last_game_date']):
            return 0
            
        last_game = pd.to_datetime(df.iloc[0]['last_game_date']).date()
        return (date - last_game).days - 1

    def team_record(self, date: date, team_abbr: str) -> dict[str, Union[int, float]]:
        """
        Get team's win-loss record up to a specific date within the season.
        """
        df = self.game_loader.load_up_to_game_season(date, team_abbr)
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
        df = self.game_loader.load_up_to_game_season(date, team_abbr)
        if len(df) == 0:
            return 0

        df = df.sort_values(['game_date', 'game_num'], ascending=True)
        
        if len(df) > num_games:
            df = df.tail(num_games)
        
        most_recent_game = df.iloc[-1]
        current_result = bool(most_recent_game['team_won'])
        streak = 1 if current_result else -1
        
        for i in range(len(df) - 2, -1, -1):
            game = df.iloc[i]
            won = bool(game['team_won'])
            
            if won == current_result:
                streak = streak + 1 if current_result else streak - 1
            else:
                break
        
        return streak
    
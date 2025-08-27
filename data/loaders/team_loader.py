import pandas as pd
from typing import Optional
from datetime import date
from pandas.core.api import DataFrame as DataFrame
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
    
    def load_for_season(self, season: int) -> DataFrame:
        return super().load_for_season(season)
    
    def load_roster(self, season: int, team: Optional[str] = None, date: Optional[date] = None) -> DataFrame:
        """Load the rosters for each date in a season. Optional filters by date and team"""
        date_filter = ""
        team_filter = ""
        params = [season]

        if team:
            team_filter = "AND team = ?"
            params.append(team)

        if date:
            if season != date.year:
                raise ValueError(f"Season and date do not match. Season{season}, date{date}")
            date_filter = "AND game_date = ?"
            params.append(date.strftime('%Y-%m-%d'))

        
        query = f"""
        SELECT 
            game_date,
            season,
            team, 
            player_name,
            normalized_player_name,
            player_id,
            position,
            status
        FROM rosters
        WHERE season = ? {team_filter} {date_filter}
        ORDER by game_date
        """

        df = self._execute_query(query, params)
        
        roster_columns = ['game_date', 'season', 'team', 'player_name', 'normalized_player_name', 'player_id', 'position', 'status']
        return self._validate_dataframe(df, roster_columns)

    def load_lineup_players(self, season: int, team: Optional[str] = None, date: Optional[date] = None) -> DataFrame:
        """Load the starting lineups for season or optionally for a single date across all teams"""
        date_filter = ""
        team_filter = ""
        params = [season]

        if team:
            team_filter = "AND team = ?"
            params.append(team)

        if date:
            if season != date.year:
                raise ValueError(f"Season and date do not match. Season{season}, date{date}")
            date_filter = "AND game_date = ?"
            params.append(date.strftime('%Y-%m-%d'))

        query = f"""
        SELECT
            game_date,
            team,
            opposing_team,
            dh,
            player_id,
            position,
            batting_order,
            season
        FROM lineup_players
        WHERE season = ? {team_filter} {date_filter}
        ORDER by game_date, dh
        """

        df = self._execute_query(query, params)
        
        lineup_columns = ['game_date', 'team', 'opposing_team', 'dh', 'player_id', 'position', 'batting_order', 'season']
        return self._validate_dataframe(df, lineup_columns)
    
    def load_pitching_matchups(self, season: int, team: Optional[str] = None, date: Optional[date] = None) -> DataFrame:
        """Load the pitching matchups for each game in a season"""
        date_filter = ""
        team_filter = ""
        params = [season]

        if team:
            team_filter = "AND team = ?"
            params.append(team)

        if date:
            if season != date.year:
                raise ValueError(f"Season and date do not match. Season{season}, date{date}")
            date_filter = "AND game_date = ?"
            params.append(date.strftime('%Y-%m-%d'))

        query = f"""
        SELECT 
            game_date,
            dh,
            team,
            opposing_team,
            team_starter_id,
            opposing_starter_id,
            season
        FROM lineups
        WHERE season = ? {team_filter} {date_filter}
        ORDER BY game_date, dh
        """

        df = self._execute_query(query, params)
        columns = ['game_date', 'dh', 'team', 'opposing_team', 'team_starter_id', 'opposing_starter_id', 'season']
        return self._validate_dataframe(df, columns)
    
    # def rest_days(self, date: date, team_abbr: str) -> int:
    #     """
    #     Calculate rest days between the target date and team's last game.
    #     """
    #     query = """
    #     SELECT MAX(game_date) as last_game_date
    #     FROM schedule
    #     WHERE (home_team = ? OR away_team = ?)
    #         AND game_date < ?
    #     """
        
    #     df = self._execute_query(query, [team_abbr, team_abbr, date.strftime('%Y-%m-%d')])
        
    #     if df.empty or pd.isna(df.iloc[0]['last_game_date']):
    #         return 0
            
    #     last_game = pd.to_datetime(df.iloc[0]['last_game_date']).date()
    #     return (date - last_game).days - 1

    # def team_record(self, date: date, team_abbr: str) -> dict[str, Union[int, float]]:
    #     """
    #     Get team's win-loss record up to a specific date within the season.
    #     """
    #     df = self.game_loader.load_up_to_game_season(date, team_abbr)
    #     if df.empty:
    #         return {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'games': 0}
        
    #     wins = df['team_won'].sum()
    #     losses = len(df) - wins
    #     win_pct = wins / len(df) if len(df) > 0 else 0.0
        
    #     return {
    #         'wins': int(wins),
    #         'losses': int(losses), 
    #         'games': len(df),
    #         'win_pct': float(win_pct)
    #     }
        
    # def game_streak(self, date: date, team_abbr: str, num_games: int = 10) -> int: 
    #     """
    #     Get current win/loss streak for a team up to a specific date.
    #     Returns positive for win streak, negative for loss streak.
    #     """
    #     df = self.game_loader.load_up_to_game_season(date, team_abbr)
    #     if len(df) == 0:
    #         return 0

    #     df = df.sort_values(['game_date', 'game_num'], ascending=True)
        
    #     if len(df) > num_games:
    #         df = df.tail(num_games)
        
    #     most_recent_game = df.iloc[-1]
    #     current_result = bool(most_recent_game['team_won'])
    #     streak = 1 if current_result else -1
        
    #     for i in range(len(df) - 2, -1, -1):
    #         game = df.iloc[i]
    #         won = bool(game['team_won'])
            
    #         if won == current_result:
    #             streak = streak + 1 if current_result else streak - 1
    #         else:
    #             break
        
    #     return streak
    
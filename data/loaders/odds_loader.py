'''
Odds loader
'''

from data.loaders.base_loader import BaseDataLoader
import pandas as pd
from datetime import date
from typing import Optional

class OddsLoader(BaseDataLoader):

    def __init__(self):
        super().__init__()
        self.columns = ['game_date', 'game_datetime', 'away_team', 'home_team', 'away_starter', 'home_starter', 
                        'away_starter_normalized', 'home_starter_normalized', 'away_score',
                        'home_score', 'winner', 'sportsbook', 'away_odds', 'home_odds', 'season']
        
    def load_for_date_range(self, start: date, end: date) -> pd.DataFrame:
        query = """
        SELECT 
            *
        FROM odds
        WHERE game_date BETWEEN ? and ?
        ORDER BY game_date
        """
        df = self._execute_query(query, [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')])
        return self._validate_dataframe(df, self.columns)

    def load_for_season(self, season: int) -> pd.DataFrame:
        """Load all odds for a season"""
        params = [season]
        columns_str = ",\n\t".join(self.columns)
        query = f"""
        SELECT 
        \t{columns_str}
        FROM odds
        WHERE season = ?
        ORDER BY game_date
        """
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.columns)

    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> pd.DataFrame:
        """
        dh must be zero in this case because table has no dh column
        """

        if dh != 0:
            raise ValueError("odds table does not have dh field.")
        
        where, params = self._time_filter(date, 0)
        query = f'''
        SELECT 
            *,
            CASE 
                WHEN home_team = ? THEN 'home'
                WHEN away_team = ? THEN 'away'
            END as team_side
        FROM odds
        WHERE ({where})
            AND (home_team = ? or away_team = ?)
        ORDER by game_date
        '''
        all_params = [team_abbr, team_abbr] + params + [team_abbr, team_abbr]
        return self._execute_query(query, all_params)
    
    def load_game_odds(self, game_date: date, away_team: str, home_team: str, away_starter: str, home_starter: str, sportsbook: Optional[str] = None):
        query = """
        SELECT
            game_date,
            game_datetime,
            away_team,
            home_team,
            away_starter,
            home_starter,
            away_score,
            home_score,
            away_odds,
            home_odds,
            winner,
            sportsbook,
            season
        FROM odds
        WHERE game_date = ?
            AND away_team = ?
            AND home_team = ?
            AND away_starter = ?
            AND home_starter = ?
            {sportsbook_filter}
        ORDER BY sportsbook
        """

        params = [game_date, away_team, home_team, away_starter, home_starter]
        if sportsbook:
            sportsbook_filter = "AND sportsbook = ?"
            params.append(sportsbook)
        else:
            sportsbook_filter = ""
        
        query = query.format(sportsbook_filter=sportsbook_filter)
        return self._execute_query(query, params)
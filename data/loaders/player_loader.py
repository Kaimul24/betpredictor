from datetime import date
from typing import Optional
from pandas.core.api import DataFrame as DataFrame
from data.loaders.base_loader import BaseDataLoader

class PlayerLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        self.batting_columns = ['player_id', 'game_date', 'team', 'batorder',
                                'pos', 'dh', 'ab', 'pa', 'ops', 'babip', 'bb_k',
                                'wrc_plus', 'woba', 'barrel_percent', 'hard_hit',
                                'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa',
                                'season']
        
        self.pitching_columns = ['player_id', 'game_date', 'team', 'dh', 'games',
                                 'gs', 'era', 'babip', 'ip', 'runs', 'k_percent',
                                 'bb_percent', 'barrel_percent', 'hard_hit', 'ev',
                                 'hr_fb', 'siera', 'fip', 'stuff', 'ifbb', 'wpa',
                                 'gmli', 'season']
        
    def _time_filter(self, date: date, dh: int = 0) -> tuple[str, list]:
        """Generate time-based filter conditions for doubleheader handling."""
        if dh <= 1:
            return "game_date < ?", [date.strftime('%Y-%m-%d')]
        else:
            return "game_date < ? OR (game_date = ? AND dh < ?)", [
                date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), dh
            ]

    def load_for_date_range(self, start: date, end: date) -> DataFrame:
        pass

    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        pass

    def load_batting_stats_for_date_range(self, start: date, end: date, team_abbr: Optional[str] = None) -> DataFrame:
        team_filter = ""
        params = [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]
        
        if team_abbr:
            team_filter = "AND team = ?"
            params.append(team_abbr)

        query = f"""
        SELECT
        \t{",\n\t".join(self.batting_columns)}
        FROM batting_stats
        WHERE game_date BETWEEN ? AND ? {team_filter}
        ORDER by game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.batting_columns)

    def load_batting_stats_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        where, params = self._time_filter(date, dh)
        query = f"""
        SELECT
        \t{",\n\t".join(self.batting_columns)}
        FROM batting_stats
        WHERE ({where}) AND team = ?
        ORDER by game_date, dh
        """
        all_params = params + [team_abbr]
        return self._execute_query(query, all_params)

    def load_pitching_stats_for_date_range(self, start: date, end: date, team_id: Optional[str] = None) -> DataFrame:
        team_filter = ""
        params = [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]

        if team_id:
            team_filter = "AND team = ?"
            params.append(team_id)
        
        query = f"""
        SELECT
        \t{",\n\t".join(self.pitching_columns)}
        FROM pitching_stats
        WHERE game_date BETWEEN ? AND ? {team_filter}
        ORDER by game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.pitching_columns)
    
    def load_pitching_stats_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        where, params = self._time_filter(date, dh)
        query = f"""
        SELECT
        \t{",\n\t".join(self.pitching_columns)}
        FROM pitching_stats
        WHERE ({where}) AND team = ?
        ORDER by game_date, dh
        """
        all_params = params + [team_abbr]
        return self._execute_query(query, all_params)

    def load_batter_stats(self, player_id: int, season: Optional[int] = None) -> DataFrame:
        year_filter = ""
        params = [player_id]

        if season:
            year_filter = "AND season = ?"
            params.append(season)

        query = f"""
        SELECT
        \t{",\n\t".join(self.batting_columns)}
        FROM batting_stats
        WHERE player_id = ? {year_filter}
        ORDER BY game_date, dh
        """

        return self._execute_query(query, params)

    def load_pitcher_stats(self, player_id: int, season: Optional[int] = None) -> DataFrame:
        year_filter = ""
        params = [player_id]

        if season:
            year_filter = "AND season = ?"
            params.append(season)

        query = f"""
        SELECT
        \t{",\n\t".join(self.pitching_columns)}
        FROM pitching_stats
        WHERE player_id = ? {year_filter}
        ORDER BY game_date, dh
        """

        return self._execute_query(query, params)
    
    def load_fielding_stats(self, season: int) -> DataFrame:
        pass
    
    def load_lineup_stats(self, team: str, date: date) -> DataFrame:
        pass

    

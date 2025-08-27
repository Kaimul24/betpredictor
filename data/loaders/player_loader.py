from datetime import date
from typing import Optional, Dict
from pandas.core.api import DataFrame as DataFrame
from data.loaders.base_loader import BaseDataLoader

class PlayerLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        self.batting_columns = ['player_id', 'mlb_id', 'game_date', 'team', 'batorder',
                                'pos', 'dh', 'ab', 'pa', 'ops', 'babip', 'bb_k',
                                'wrc_plus', 'woba', 'barrel_percent', 'hard_hit',
                                'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa',
                                'season']
        
        self.pitching_columns = ['player_id', 'mlb_id', 'name', 'normalized_player_name', 
                                 'game_date', 'team', 'dh', 'games','gs', 'era',
                                'babip', 'ip', 'tbf', 'bip', 'runs', 'k_percent',
                                 'bb_percent', 'barrel_percent', 'hard_hit', 'ev',
                                 'hr_fb', 'siera', 'fip', 'stuff', 'iffb', 'wpa',
                                 'gmli',  'fa_percent', 'fc_percent', 'si_percent',
                                 'fa_velo', 'fc_velo', 'si_velo' ,'season'
                                 ]
        
        self.fielding_columns = ['name', 'normalized_player_name', 'season', 'month',
                                 'frv', 'total_innings', 'innings_c', 'innings_1B', 
                                 'innings_2B', 'innings_3B', 'innings_SS', 'innings_LF',
                                 'innings_CF', 'innings_RF',
                                 ]

    def load_for_date_range(self, start: date, end: date) -> DataFrame:
        raise NotImplementedError("Use load_<batting/pitching_stats>_for_date_range instead.")

    def load_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        raise NotImplementedError("Use load_<batting/pitching_stats>up_to_game instead.")

    def load_for_season(self, season: int) -> DataFrame:
        raise NotImplementedError("Use load_for_season<batter/pitcher> instead.")

    def load_for_season_batter(self, season: int) -> DataFrame:
        params = [season]
        columns_str = ",\n\t".join(self.batting_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM batting_stats
        WHERE season = ?
        ORDER BY game_date, dh
        """
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.batting_columns)
    
    def load_for_season_pitcher(self, season: int) -> DataFrame:
        params = [season]
        columns_str = ",\n\t".join(self.pitching_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM pitching_stats
        WHERE season = ?
        ORDER BY game_date, dh
        """
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.pitching_columns)


    def load_batting_stats_for_date_range(self, start: date, end: date, team_abbr: Optional[str] = None) -> DataFrame:
        team_filter = ""
        params = [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]
        
        if team_abbr:
            team_filter = "AND team = ?"
            params.append(team_abbr)

        columns_str = ",\n\t".join(self.batting_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM batting_stats
        WHERE game_date BETWEEN ? AND ? {team_filter}
        ORDER by game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.batting_columns)

    def load_batting_stats_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        where, params = self._time_filter(date, dh)
        columns_str = ",\n\t".join(self.batting_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM batting_stats
        WHERE ({where}) AND team = ?
        ORDER by game_date, dh
        """
        all_params = params + [team_abbr]
        df = self._execute_query(query, all_params)
        return self._validate_dataframe(df, self.batting_columns)

    def load_pitching_stats_for_date_range(self, start: date, end: date, team_id: Optional[str] = None) -> DataFrame:
        team_filter = ""
        params = [start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')]

        if team_id:
            team_filter = "AND team = ?"
            params.append(team_id)
        
        columns_str = ",\n\t".join(self.pitching_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM pitching_stats
        WHERE game_date BETWEEN ? AND ? {team_filter}
        ORDER by game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.pitching_columns)
    
    def load_pitching_stats_up_to_game(self, date: date, team_abbr: str, dh: int = 0) -> DataFrame:
        where, params = self._time_filter(date, dh)
        columns_str = ",\n\t".join(self.pitching_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM pitching_stats
        WHERE ({where}) AND team = ?
        ORDER by game_date, dh
        """
        all_params = params + [team_abbr]
        df = self._execute_query(query, all_params)
        return self._validate_dataframe(df, self.pitching_columns)

    def load_batter_stats(self, player_id: int, season: Optional[int] = None) -> DataFrame:
        year_filter = ""
        params = [player_id]

        if season:
            year_filter = "AND season = ?"
            params.append(season)

        columns_str = ",\n\t".join(self.batting_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM batting_stats
        WHERE player_id = ? {year_filter}
        ORDER BY game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.batting_columns)

    def load_pitcher_stats(self, player_id: int, season: Optional[int] = None) -> DataFrame:
        year_filter = ""
        params = [player_id]

        if season:
            year_filter = "AND season = ?"
            params.append(season)

        columns_str = ",\n\t".join(self.pitching_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM pitching_stats
        WHERE player_id = ? {year_filter}
        ORDER BY game_date, dh
        """

        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.pitching_columns)
    
    def load_fielding_stats(self, season: int) -> DataFrame:
        """Loads fielding stats by season"""
        params = [season]
        columns_str = ",\n\t".join(self.fielding_columns)
        query = f"""
        SELECT
        \t{columns_str}
        FROM fielding
        WHERE season = ?
        ORDER BY season, month
        """
        df = self._execute_query(query, params)
        return self._validate_dataframe(df, self.fielding_columns)
    
    def get_avg_batting_stats(self, season: int, stats: str = None) -> Dict[str, float]:
        """Returns the league average batting stats. Default is all batting stats"""
        pass
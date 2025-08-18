"""
Handles computation of batting stats features.
"""
import pandas as pd
from typing import List
from data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from data.loaders.player_loader import PlayerLoader
from src.config import FEATURES_CACHE_PATH
import logging, os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BATTING_CACHE_PATH = os.getenv("rolling_batting_features_cache")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data.database import get_database_manager

class BattingFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame):
        super().__init__(season, data)
        self.rolling_windows = [25,14,9,5,3]
        self.rolling_metrics = ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 
                               'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb',
                               'baserunning', 'wraa', 'wpa']
    
    def calculate_all_player_rolling_stats(self, force_calculate: bool = False) -> DataFrame:
        """Calculate and store rolling stats for all players and store in database"""
        cache_path = Path(FEATURES_CACHE_PATH / BATTING_CACHE_PATH.format(self.season))

        if cache_path.exists()and not force_calculate:
            logger.info(f" Found cached batter rolling stats for {self.season}")
            batting_features = pd.read_parquet(cache_path)
            return batting_features
        elif force_calculate:
            logger.info(f" Recaluclating batter rolling stats...")
            
        logger.info(f" No cached batter rolling stats found for {self.season}")
        db = get_database_manager()

        players = """
        SELECT DISTINCT player_id
        FROM batting_stats
        WHERE season = ?
        """

        players = db.execute_read_query(players, (self.season,))
        logger.info(f" Calculating rolling stats for {len(players)} players in {self.season}")
        all_players = []
        batch_size = 50
        for i in range(0, len(players), batch_size):
            batch = players[i:i+batch_size]
            batch_result = self._process_player_batch([p['player_id'] for p in batch])
            if not batch_result.empty:
                all_players.append(batch_result)
            logger.info(f" Processed batch {i//batch_size + 1}/{(len(players) + batch_size - 1)//batch_size}")

        if not all_players:
            logger.warning(f" No batting data found for season {self.season}")
            return pd.DataFrame()

        df = pd.concat(all_players, ignore_index=True)
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_parquet(cache_path, index=True)
            logger.info(f" Successfully cached batting rolling stats to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache batting rolling stats: {e}")

        return df


    def _process_player_batch(self, player_ids: List[str]) -> DataFrame:
        """Process a batch of players and insert their rolling stats"""
        db = get_database_manager()
        all_rolling_stats = []
        
        with db.get_reader_connection() as conn:
            for player_id in player_ids:

                player_query = """
                SELECT player_id, game_date, team, dh, ops, wrc_plus, woba, babip, bb_k,
                       barrel_percent, hard_hit, ev, iso, gb_fb, baserunning, wraa, wpa, pa
                FROM batting_stats 
                WHERE player_id = ? AND season = ?
                ORDER BY game_date, dh
                """
                
                player_data = pd.read_sql_query(
                    player_query, 
                    conn,
                    params=[player_id, self.season]
                )
                
                if not player_data.empty:
                    player_rolling = self._calculate_rolling_window_for_player(player_data)
                    if not player_rolling.empty:
                        all_rolling_stats.append(player_rolling)
        

        if all_rolling_stats:
            combined_stats = pd.concat(all_rolling_stats, ignore_index=True)
            return combined_stats
        else:
            return pd.DataFrame()


    def _calculate_rolling_window_for_player(self, player_data: DataFrame) -> DataFrame:
        """Calculate rolling stats for a single player across all windows"""
        all_results = []
        
        player_data['game_date'] = pd.to_datetime(player_data['game_date'])
        player_data = player_data.sort_values(['game_date', 'dh'])
        
        for window in self.rolling_windows:
            rolling_data = player_data.copy()
            
            for metric in self.rolling_metrics:
                if metric in rolling_data.columns:

                    if metric in ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 'ev', 'iso']:

                        weighted_sum = (rolling_data[metric] * rolling_data["pa"]).rolling(
                            window=window, min_periods=1
                        ).sum()
                        
                        total_pa = rolling_data["pa"].rolling(
                            window=window, min_periods=1
                        ).sum()
                        
                        rolling_values = weighted_sum / total_pa
                    else:

                        rolling_values = rolling_data[metric].rolling(
                            window=window, min_periods=1
                        ).mean()
                    
                    rolling_data[f"{metric}_rolling"] = rolling_values.shift(1)
            
            rolling_data['window_size'] = window
            rolling_data['games_in_window'] = rolling_data['pa'].rolling(
                window=window, min_periods=1
            ).count().shift(1)

            rolling_data['total_pa_in_window'] = rolling_data['pa'].rolling(
                window=window, min_periods=1
            ).sum().shift(1)

            rolling_data['season'] = self.season
            
            all_results.append(rolling_data)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
    


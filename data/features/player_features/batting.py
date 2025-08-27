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

    def __init__(self, season: int, data: DataFrame, force_recreate: bool = False):
        super().__init__(season, data)
        self.force_recreate = force_recreate
        self.rolling_windows = [25,14,9,5,3,0]
        self.rolling_metrics = ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 
                               'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb',
                               'baserunning', 'wraa', 'wpa']
        
    def load_features(self):
        return self.calculate_all_player_rolling_stats()
    
    def calculate_all_player_rolling_stats(self) -> DataFrame:
        """Calculate rolling stats for all batters"""
        cache_path = Path(FEATURES_CACHE_PATH / BATTING_CACHE_PATH.format(self.season))

        if cache_path.exists()and not self.force_recreate:
            logger.info(f" Found cached batter rolling stats for {self.season}")
            batting_features = pd.read_parquet(cache_path)
            return batting_features
        elif self.force_recreate:
            logger.info(f" Recaluclating batter rolling stats...")
            
        logger.info(f" No cached batter rolling stats found for {self.season}")
        players = self.data['player_id'].unique().copy() 
        logger.info(f" Calculating rolling stats for {len(players)} players in {self.season}")

        all_players = []
        batch_size = 50
        for i in range(0, len(players), batch_size):
            batch = players[i:i+batch_size]
            batch_result = self._process_player_batch([p for p in batch])
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
        """Process a batch of players and return their rolling stats"""
        all_rolling_stats = []

        for player_id in player_ids:
            
            player_data = self.data[self.data['player_id'] == player_id].copy()
            
            if not player_data.empty:
                
                required_cols = ['player_id', 'game_date', 'team', 'dh', 'ops', 'wrc_plus', 'woba', 'babip', 'bb_k',
                               'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb', 'baserunning', 'wraa', 'wpa', 'pa']
                player_data = player_data[required_cols].sort_values(['game_date', 'dh'])
                
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

        def rolling_vs_expanding(window_):
            return window_.expanding(min_periods=1) if season_window else window_.rolling(window=window, min_periods=1)

        all_results = []
        
        player_data['game_date'] = pd.to_datetime(player_data['game_date'])
        player_data = player_data.sort_values(['game_date', 'dh'])
        
        for window in self.rolling_windows:
            season_window = (window == 0)

            data = player_data.copy()
            
            for metric in self.rolling_metrics:
                if metric in data.columns:  
                    # PA weighted metrics
                    if metric in ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 'ev', 'iso']:

                        stat = rolling_vs_expanding(data[metric] * data['pa']).sum()
                        
                        total_pa = rolling_vs_expanding(data['pa']).sum()
                        
                        rolling_values = stat / total_pa
                    else:
                        # Simple averages
                        rolling_values = rolling_vs_expanding(data[metric]).mean()
                    
                    data[f"{metric}_rolling"] = rolling_values.shift(1)
            
            data['window_size'] = window
            data['games_in_window'] = rolling_vs_expanding(data['pa']).count().shift(1)
            data['total_pa_in_window'] = rolling_vs_expanding(data['pa']).sum().shift(1)
            data['season'] = self.season
            
            all_results.append(data)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
    


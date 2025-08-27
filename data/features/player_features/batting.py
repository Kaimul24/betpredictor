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
        super().__init__(season, data, force_recreate)
        self.rolling_windows = [25,14,9,5,3,0]
        self.rolling_metrics = ['ops', 'wrc_plus', 'woba', 'babip', 'bb_k', 
                               'barrel_percent', 'hard_hit', 'ev', 'iso', 'gb_fb',
                               'baserunning', 'wraa', 'wpa']
        
    def load_features(self):
        return self.rolling_batting_stats()
    
    def rolling_batting_stats(self) -> DataFrame:
        cache_path = Path(FEATURES_CACHE_PATH / BATTING_CACHE_PATH.format(self.season))
        
        if cache_path.exists() and not self.force_recreate:
            logger.info(f" Found cached batter rolling stats for {self.season}")
            batting_features = pd.read_parquet(cache_path)
            return batting_features
        elif self.force_recreate:
            if cache_path.exists():
                try:
                    logger.info( f"Removing old batter rolling_stats...")
                    os.remove(cache_path)
                except OSError as e:
                    logger.error(f"Error deleting file '{cache_path}': {e}")
                
            logger.info(f" Caluclating batter rolling stats...")

        df = self.data.copy()
        df.sort_values(['player_id', 'game_date', 'dh'])

        prior_specs = {
            # “true rate” / index-style metrics (weight by PA unless noted)
            "prior_woba":            ("woba",           "pa"),
            "prior_ops":             ("ops",            "pa"),
            "prior_wrc_plus":        ("wrc_plus",       "pa"),   # will be ≈100 by construction if you pass one season
            "prior_babip":           ("babip",          "bip"),   # ideally BIP; PA is a solid proxy if BIP not tracked
            "prior_iso":             ("iso",            "ab"),   # ISO is AB-based
            "prior_bb_k":            ("bb_k",           "pa"),   # ratio; PA-weighted grand mean is robust in practice

            # batted-ball quality (ideally weight by BBE; if you don’t have it, PA is a decent proxy)
            "prior_barrel_percent":  ("barrel_percent", "bip"),
            "prior_hard_hit":        ("hard_hit",       "bip"),
            "prior_ev":              ("ev",             "bip"),   # average EV

            # batted-ball shape
            "prior_gb_fb":           ("gb_fb",          "bip"),   # ideally GB+FB count; PA as proxy

            # run value / leverage-y things (center near 0)
            "prior_baserunning":  ("baserunning",    "pa"),   # BsR per PA tends toward ~0 as a neutral prior
            "prior_wraa":         ("wraa",           "pa"),   # wRAA per PA, neutral ≈ 0
            "prior_wpa":          ("wpa",            "pa"),   # WPA per PA, neutral ≈ 0
        }

        shrinkage_weights_cols = ["pa", "ab", "bip"]
    
        ewm_cols = {
            # Overall quality
            "woba":        ("woba", "pa",  "prior_woba",        150, True),
            "ops":         ("ops",  "pa",  "prior_ops",         150, True),
            "wrc_plus":    ("wrc_plus","pa","prior_wrc_plus",   220, True),  # index-y -> stronger shrink

            # Contact & power
            "iso":         ("iso",  "ab",  "prior_iso",         160, True),  # AB-based
            "babip":       ("babip","bip", "prior_babip",       320, True),  # very noisy, needs larger k

            # Batted-ball quality — BIP-weighted
            "barrel_pct":  ("barrel_percent","bip","prior_barrel_percent",  80, True),
            "hard_hit":    ("hard_hit", "bip","prior_hard_hit",        110, True),
            "ev":          ("ev",      "bip","prior_ev",                60, True),

            # Batted-ball shape (ratio-y => heavier shrink)
            "gb_fb":       ("gb_fb",   "bip","prior_gb_fb",            180, True),

            # Discipline proxy (ratio-y => heavier shrink)
            "bb_k":        ("bb_k",    "pa", "prior_bb_k",             200, True),

            # Run-value style (very noisy, strong shrinkage)
            "baserunning": ("baserunning","pa","prior_baserunning",    220, True),
            "wraa":        ("wraa",    "pa", "prior_wraa",             260, True),
            "wpa":         ("wpa",     "pa", "prior_wpa",              400, True),
        }

        result, priors = BaseFeatures.compute_rolling_stats(
            player_data=df,                 # season-scoped
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=["player_id","team","game_date","dh","season"],
            halflives=(3, 10, 25)  # or (3, 10, 30)
        )
        
        try:
            result.to_parquet(cache_path, index=True)
            logger.info(f" Successfully cached batting rolling stats to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache batting rolling stats: {e}")

        return result
    
    def calculate_all_player_rolling_stats(self) -> DataFrame:
        """Calculate rolling stats for all batters"""
        cache_path = Path(FEATURES_CACHE_PATH / BATTING_CACHE_PATH.format(self.season))

        if cache_path.exists() and not self.force_recreate:
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
    


"""
Handles computation of batting stats features.
"""
import pandas as pd
from typing import List
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from src.data.loaders.player_loader import PlayerLoader
from src.config import FEATURES_CACHE_PATH
import logging, os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BATTING_CACHE_PATH = os.getenv("rolling_batting_features_cache")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BattingFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame, force_recreate: bool = False):
        super().__init__(season, data, force_recreate)
        
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
                    logger.info(f" Removing old batter rolling_stats...")
                    os.remove(cache_path)
                except OSError as e:
                    logger.error(f"Error deleting file '{cache_path}': {e}")
                
            logger.info(f" Calculating batter rolling stats...")

        df = self.data.copy()
        df.sort_values(['player_id', 'game_date', 'dh'])

        prior_specs = {
            "prior_woba":            ("woba",           "pa"),
            "prior_ops":             ("ops",            "pa"),
            "prior_wrc_plus":        ("wrc_plus",       "pa"),
            "prior_babip":           ("babip",          "bip"),  
            "prior_iso":             ("iso",            "ab"),
            "prior_bb_k":            ("bb_k",           "pa"), 
            "prior_barrel_percent":  ("barrel_percent", "bip"),
            "prior_hard_hit":        ("hard_hit",       "bip"),
            "prior_ev":              ("ev",             "bip"),
            "prior_gb_fb":           ("gb_fb",          "bip"),
            "prior_baserunning":    ("baserunning",     "pa"),
            "prior_wraa":           ("wraa",            "pa"), 
            "prior_wpa":            ("wpa",             "pa"),
        }

        shrinkage_weights_cols = ["pa", "ab", "bip"]
    
        ewm_cols = {
            "woba":             ("woba", "pa",  "prior_woba",                    150, True),
            "ops":              ("ops",  "pa",  "prior_ops",                     150, True),
            "wrc_plus":         ("wrc_plus","pa","prior_wrc_plus",               220, True),
            "iso":              ("iso",  "ab",  "prior_iso",                     160, True),
            "babip":            ("babip","bip", "prior_babip",                   320, True),
            "barrel_percent":   ("barrel_percent","bip","prior_barrel_percent",  80, True),
            "hard_hit":         ("hard_hit", "bip","prior_hard_hit",             110, True),
            "ev":               ("ev",      "bip","prior_ev",                    60, True),
            "gb_fb":            ("gb_fb",   "bip","prior_gb_fb",                 180, True),
            "bb_k":             ("bb_k",    "pa", "prior_bb_k",                  200, True),
            "baserunning":      ("baserunning","pa","prior_baserunning",         220, True),
            "wraa":             ("wraa",    "pa", "prior_wraa",                  260, True),
            "wpa":              ("wpa",     "pa", "prior_wpa",                   400, True),
        }

        result, priors = BaseFeatures.compute_rolling_stats(
            player_data=df,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=ewm_cols,
            preserve_cols=["player_id", "mlb_id", "pos", "team", "game_date", "dh", "season", "ab", "pa"],
            halflives=(3, 10, 25)
        )
        
        try:
            result.to_parquet(cache_path, index=True)
            logger.info(f" Successfully cached batting rolling stats to {cache_path}")
        except Exception as e:
            logger.error(f" Failed to cache batting rolling stats: {e}")

        return result
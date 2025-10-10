"""
Abstract base class for all features
"""

from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class BaseFeatures(ABC):

    def __init__(self, season: int, data: DataFrame, force_recreate: bool = False):
        self.season = season
        self.data = data
        self.force_recreate = force_recreate

    @abstractmethod
    def load_features(self) -> DataFrame:
        pass

    @staticmethod
    def compute_ewm(player_stats: pd.Series, weight_by: pd.Series, by: pd.Series, half_life: int, val_is_rate=False) -> pd.Series:
        """Compute EWM weighted mean per player. Can be rate weighted or a simple mean"""
        s = player_stats.fillna(0)

        w = weight_by.fillna(0)
        num = (s * w) if val_is_rate else s

        num_ewm = num.groupby(by).transform(
            lambda x: x.ewm(halflife=half_life, adjust=False, min_periods=1).mean()
        )

        den_ewm = w.groupby(by).transform(
            lambda x: x.ewm(halflife=half_life, adjust=False, min_periods=1).mean()
        )

        result = num_ewm / den_ewm.replace(0, np.nan)
        return result.groupby(by).shift(1)

    @staticmethod
    def cumsum_shift(metric: pd.Series, by: pd.Series) -> pd.Series:
        return metric.fillna(0).groupby(by).cumsum().groupby(by).shift(1)

    @staticmethod
    def shrinkage(N: pd.Series, k: pd.Series, stat: pd.Series, stat_prior: float) -> pd.Series:
        N = N.fillna(0)
        den = (N + k)

        result = (N / den) * stat + (k / den) * stat_prior

        result = result.where(N > 0, stat_prior)
        return result
    
    @staticmethod
    def _weighted_mean(stat: pd.Series, weight: pd.Series) -> float:
        stat = stat.fillna(0)
        weight = weight.fillna(0)

        num = (stat * weight).sum()
        den = weight.sum()

        return num / den if den > 0 else np.nan
    
    @staticmethod
    def expanding_weighted_mean(
                val: pd.Series,
                weight: pd.Series,
                by: pd.Series,
                val_is_rate: bool = True) -> Tuple[pd.Series, pd.Series]:
        """
        Season-to-date (expanding) weighted mean and N (cumulative weight) per group.
        """
        v = val.fillna(0)
        w = weight.fillna(0)
        num = (v * w) if val_is_rate else v
        den = w

        num_cum = num.groupby(by).cumsum()
        den_cum = den.groupby(by).cumsum()

        rate = num_cum / den_cum.replace(0, np.nan)

        rate = rate.groupby(by).shift(1)
        den_cum = den_cum.groupby(by).shift(1)
        return rate, den_cum
    
    @staticmethod
    def compute_weighted_priors(data: DataFrame, prior_specs) -> Tuple[DataFrame, Dict[str, float]]:

        df = data.copy()

        priors = {key: BaseFeatures._weighted_mean(df[val], df[weight]) 
                  for key, (val, weight) in prior_specs.items()}
        
        for k, v in priors.items():
            df[k] = v

        return df, priors
    
    @staticmethod
    def compute_rolling_stats(
                        data: DataFrame,
                        prior_specs: Dict[str, Tuple[str, str]], 
                        shrinkage_weights_cols: List[str], 
                        ewm_cols: Dict[str, Tuple[str, str, str, int, bool]],
                        preserve_cols: List[str],
                        by: pd.Series = pd.Series([]),
                        halflives=(3, 8, 20)) -> Tuple[DataFrame, Dict]:

        df = data.copy()
        
        if by.empty and not df.empty:
            by = df['player_id']
        
        df, priors = BaseFeatures.compute_weighted_priors(
            df, prior_specs
        )

        shrink_weights = {
            key: BaseFeatures.cumsum_shift(df[key], by) for key in shrinkage_weights_cols
        }

        result = df[preserve_cols].copy()

        for name, (val_col, denom_key, prior_col, k, val_is_rate) in ewm_cols.items():
            
            val = df[val_col]
            
            den = df[denom_key]

            season_stats, N_season = BaseFeatures.expanding_weighted_mean(val, den, by, val_is_rate=val_is_rate)
            
            result[f'{name}_season'] = BaseFeatures.shrinkage(N_season, k, season_stats, df[prior_col])

            for hl in halflives:
                
                rate_hl = BaseFeatures.compute_ewm(val, den, by, half_life=hl, val_is_rate=val_is_rate)

                result[f'{name}_ewm_h{hl}'] = BaseFeatures.shrinkage(shrink_weights[denom_key], k, rate_hl, df[prior_col])


        # for name, (_val_col, _denom_key, prior_col, _k, _is_rate) in ewm_cols.items():
        #     col = f"{name}_season"
        #     if col in result.columns and prior_col in df.columns:
        #         result[col] = result[col].fillna(df[prior_col])
        #     for hl in halflives:
        #         col = f"{name}_ewm_h{hl}"
        #         if col in result.columns and prior_col in df.columns:
        #             result[col] = result[col].fillna(df[prior_col])

        if 'game_date' in df.columns:
            result['last_app_date'] = df.groupby(by)['game_date'].shift(1)
        
        return result, priors
    
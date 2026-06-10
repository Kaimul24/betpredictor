"""
Abstract base class for all features
"""

from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame
from typing import List, Dict, Tuple, Optional
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
    def compute_weighted_priors(
                    data: DataFrame, 
                    prior_specs: Dict[str, Tuple[str, str]],
                    prior_data: Optional[DataFrame] = None,
                    fallback_prior_data: Optional[DataFrame] = None,
                    min_samples: Optional[Dict[str, float]] = None,
                ) -> Tuple[DataFrame, Dict[str, float]]:

        df = data.copy()

        if prior_data is not None and not prior_data.empty:
            return BaseFeatures.compute_player_priors(
                df,
                prior_specs,
                prior_data,
                fallback_prior_data=fallback_prior_data,
                min_samples=min_samples,
            )

        priors = {key: BaseFeatures._weighted_mean(df[val], df[weight]) 
                  for key, (val, weight) in prior_specs.items()}
        
        for k, v in priors.items():
            df[k] = v

        return df, priors

    @staticmethod
    def compute_player_priors(
        data: DataFrame,
        prior_specs: Dict[str, Tuple[str, str]],
        prior_data: DataFrame,
        fallback_prior_data: Optional[DataFrame] = None,
        min_samples: Optional[Dict[str, float]] = None,
    ) -> Tuple[DataFrame, Dict[str, float]]:
        df = data.copy()
        min_samples = min_samples or {}
        priors = {}

        for prior_col, (val_col, weight_col) in prior_specs.items():
            league_prior = BaseFeatures._prior_league_average(df, prior_data, val_col, weight_col)
            priors[prior_col] = league_prior

            player_priors = BaseFeatures._player_prior_map(
                prior_data,
                val_col,
                weight_col,
                min_samples.get(weight_col, 0),
            )
            prior_values = df["player_id"].map(player_priors)

            if fallback_prior_data is not None and not fallback_prior_data.empty:
                fallback_priors = BaseFeatures._player_prior_map(
                    fallback_prior_data,
                    val_col,
                    weight_col,
                    min_samples.get(weight_col, 0),
                )
                prior_values = prior_values.fillna(df["player_id"].map(fallback_priors))

            df[prior_col] = prior_values.fillna(league_prior)

        return df, priors

    @staticmethod
    def _prior_league_average(
        current_data: DataFrame,
        prior_data: DataFrame,
        val_col: str,
        weight_col: str,
    ) -> float:
        source = prior_data if {val_col, weight_col}.issubset(prior_data.columns) else current_data
        if not {val_col, weight_col}.issubset(source.columns):
            return np.nan

        prior = BaseFeatures._weighted_mean(source[val_col], source[weight_col])
        if pd.isna(prior) and {val_col, weight_col}.issubset(current_data.columns):
            prior = BaseFeatures._weighted_mean(current_data[val_col], current_data[weight_col])
        return prior

    @staticmethod
    def _player_prior_map(
        prior_data: DataFrame,
        val_col: str,
        weight_col: str,
        min_sample: float,
    ) -> pd.Series:
        if prior_data.empty or "player_id" not in prior_data.columns:
            return pd.Series(dtype=float)
        if not {val_col, weight_col}.issubset(prior_data.columns):
            return pd.Series(dtype=float)

        source = prior_data[["player_id", val_col, weight_col]].copy()
        source[val_col] = pd.to_numeric(source[val_col], errors="coerce").fillna(0)
        source[weight_col] = pd.to_numeric(source[weight_col], errors="coerce").fillna(0)
        source["_weighted_value"] = source[val_col] * source[weight_col]

        grouped = source.groupby("player_id", observed=True).agg(
            weighted_value=("_weighted_value", "sum"),
            weight=(weight_col, "sum"),
        )
        priors = grouped["weighted_value"] / grouped["weight"].replace(0, np.nan)
        return priors.where(grouped["weight"] >= min_sample).dropna()
    
    @staticmethod
    def compute_rolling_stats(
        data: DataFrame,
        prior_specs: Dict[str, Tuple[str, str]], 
        shrinkage_weights_cols: List[str], 
        ewm_cols: Dict[str, Tuple[str, str, str, int, bool]],
        preserve_cols: List[str],
        by: pd.Series = pd.Series([]),
        halflives=(3, 8, 20),
        prior_data: Optional[DataFrame] = None,
        fallback_prior_data: Optional[DataFrame] = None,
        prior_min_samples: Optional[Dict[str, float]] = None,
    ) -> Tuple[DataFrame, Dict]:

        df = data.copy()
        
        if by.empty and not df.empty:
            by = df['player_id']
        
        df, priors = BaseFeatures.compute_weighted_priors(
            df,
            prior_specs,
            prior_data=prior_data,
            fallback_prior_data=fallback_prior_data,
            min_samples=prior_min_samples,
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

        if 'game_date' in df.columns:
            result['last_app_date'] = df.groupby(by)['game_date'].shift(1)
        
        return result, priors
    
    @staticmethod
    def _add_matchup_cols_diff_same_base(df: DataFrame, cols: List[str], ewm_cols: List[str]) -> Dict[str, pd.Series]:
        result = {}
        for col in cols:
            for c in ewm_cols:
                result[f"home_away_{col}_{c}_diff"] = df[f"home_{col}_{c}"] - df[f"away_{col}_{c}"]

        return result
    
    @staticmethod
    def _add_matchup_cols_diff_base(df: DataFrame, col1: List[str], col2: List[str], col1_ewm_cols: List[str], col2_ewm_cols: List[str]) -> Dict[str, pd.Series]:
        if len(col1) != len(col2) or len(col1_ewm_cols) != len(col2_ewm_cols):
            raise ValueError("Col1 and Col2 must be same length.")

        result = {}
        for c1, c2 in zip(col1, col2):
            for e1, e2 in zip(col1_ewm_cols, col2_ewm_cols):
                result[f"home_away_{c1}_{c2}_{e1}_diff"] = df[f"home_{c1}_{e1}"] - df[f"away_{c2}_{e2}"]

        return result

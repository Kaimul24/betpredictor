"""
Handles computation of pitching stats features. Classifies each pitcher as starter/reliever.
Also handles bullpen status
"""

import pandas as pd
import numpy as np
import logging, os
from src.data.features.base_feature import BaseFeatures
from pandas.core.api import DataFrame as DataFrame
from typing import Tuple
from src.config import FEATURES_CACHE_PATH

from src.data.loaders.team_loader import TeamLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
PITCHING_CACHE_PATH = os.getenv("rolling_pitching_features_cache")

class PitchingFeatures(BaseFeatures):

    def __init__(self, season: int, data: DataFrame, force_recreate: bool = False) -> None:
        super().__init__(season, data, force_recreate)
        self.pitching_matchups = TeamLoader().load_pitching_matchups(self.season)

    def load_features(self) -> DataFrame:
        "Loads all pitching features"
        cache_path = Path(FEATURES_CACHE_PATH / PITCHING_CACHE_PATH.format(self.season))
        
        if cache_path.exists() and not self.force_recreate:
            logger.info(f" Found cached pitching rolling stats for {self.season}")
            pitching_features = pd.read_parquet(cache_path)
            return pitching_features
        
        elif self.force_recreate:
            if cache_path.exists():
                try:
                    logger.info( f"Removing old pitcher rolling stats...")
                    os.remove(cache_path)
                except OSError as e:
                    logger.error(f"Error deleting file '{cache_path}': {e}")
            else:
                logger.info(f" No cached pitching stats for {self.season}")

        logger.info(f" Calculating pitcher rolling stats...")
        logger.info(" Loading roster...")
        roster_data = TeamLoader().load_roster(self.season)
        pitching_matchups = self.pitching_matchups.copy()
        pitching_rosters = roster_data[(roster_data['position'] == 'P') | (roster_data['position'] == 'TWP')]

        pitching_matchups['team_starter_id'] = pitching_matchups['team_starter_id'].astype('Int64')
        pitching_matchups['opposing_starter_id'] = pitching_matchups['opposing_starter_id'].astype('Int64')

        na_mask = pitching_matchups['team_starter_id'].isna()
        if na_mask.any():
            logger.info(f"Found {na_mask.sum()} missing team_starter_id values, filling from opposing team records...")
            
            lookup_df = pitching_matchups[~na_mask].copy()
            lookup_dict = lookup_df.set_index(['game_date', 'dh', 'opposing_team'])['opposing_starter_id'].to_dict()
            
            for idx in pitching_matchups[na_mask].index:
                row = pitching_matchups.loc[idx]
                lookup_key = (row['game_date'], row['dh'], row['team'])
                if lookup_key in lookup_dict:
                    pitching_matchups.loc[idx, 'team_starter_id'] = lookup_dict[lookup_key]
        
        self.pitching_matchups = pitching_matchups

        pitching_stats = self._classify_pitcher_type()
        pitching_stats = self._is_opener(pitching_stats)
        pitching_stats['player_id'] = pitching_stats['player_id'].astype("Int64")

        reliver_rolling_stats, starters_rolling_stats, rl_priors, st_priors = self._rolling_pitching_stats(pitching_stats)

        logger.info(f" Reliever NaN rows (_rolling_pitching_stats): {reliver_rolling_stats.isna().any(axis=1).sum()}")
        logger.info(f" Starter NaN rows (_rolling_pitching_stats): {starters_rolling_stats.isna().any(axis=1).sum()}")

        team_starter_stats = pd.merge(
            self.pitching_matchups,
            starters_rolling_stats,
            left_on=['game_date', 'dh', 'team_starter_id'],
            right_on=['game_date', 'dh', 'player_id'],
            how='left',
            suffixes=('', '_s')
        )
        
        cols = ['mlb_id', 'season']
        team_starter_stats.drop(columns=[col for col in team_starter_stats if col.endswith('_s') or col in cols], inplace=True)
        

        all_matchups_with_stats = pd.merge(
            team_starter_stats,
            starters_rolling_stats,
            left_on=['game_date', 'dh', 'opposing_starter_id', 'opposing_team'],
            right_on=['game_date', 'dh', 'player_id', 'team'],
            how='left',
            suffixes=('_team_starter', '_opposing_starter')
        )
    

        all_matchups_with_stats.drop(columns=['team_opposing_starter'], inplace= True)
        all_matchups_with_stats.rename(columns={'team_team_starter': 'team'}, inplace=True)
        
        reliver_rolling_stats = reliver_rolling_stats.rename(columns={'player_id': 'fg_id', 'mlb_id': 'player_id'})
        starters_rolling_stats = starters_rolling_stats.rename(columns={'player_id': 'fg_id', 'mlb_id': 'player_id'})
        
        starters_map = self.pitching_matchups[['game_date', 'team', 'team_starter_id']].drop_duplicates()
        
        rosters_plus = self._fill_missing_rosters(pitching_rosters, starters_map)

        rosters_plus['player_id'] = rosters_plus['player_id'].astype('Int64')
        rosters_plus['team_starter_id'] = rosters_plus['team_starter_id'].astype('Int64')
        bullpen_roster = rosters_plus[rosters_plus['player_id'] != rosters_plus['team_starter_id']].copy()
        
        reliever_stats = reliver_rolling_stats.drop(columns=['dh'], errors='ignore').copy()
        
        bullpen_roster['player_id'] = bullpen_roster['player_id'].astype('int64')
        reliever_stats['player_id'] = reliever_stats['player_id'].astype('int64')
        
        by_key = 'player_id'
        on_key = 'game_date'
        sort_keys = [on_key] + [by_key]

        bullpen_roster = bullpen_roster.sort_values(sort_keys, kind='mergesort').reset_index(drop=True)
        reliever_stats = reliever_stats.sort_values(sort_keys, kind='mergesort').reset_index(drop=True)

        assert bullpen_roster[on_key].is_monotonic_increasing, "left game_date not globally sorted"
        assert reliever_stats[on_key].is_monotonic_increasing, "right game_date not globally sorted"

        bullpen_roster_with_stats = pd.merge_asof(
            left=bullpen_roster,
            right=reliever_stats,
            by=by_key,
            left_on=on_key,
            right_on=on_key,
            direction='backward',
            allow_exact_matches=False,
            suffixes=("", "_rel")
        )

        bullpen_roster_with_stats.drop(columns=[col for col in bullpen_roster_with_stats.columns
                                                if col.endswith('_rel')], inplace=True)
        
        bullpen_roster_with_stats['game_date'] = pd.to_datetime(bullpen_roster_with_stats['game_date'])
        
        value_cols = [c for c in bullpen_roster_with_stats.columns 
              if any(s in c for s in ['_season', '_ewm_h3', '_ewm_h8', '_ewm_h20', '_li'])
              or c in ['fip', 'siera', 'stuff', 'gmli', 'wpa_li']]

        usage_df = self._compute_bullpen_usage(bullpen_roster_with_stats)
        
        bullpen_agg = (
            bullpen_roster_with_stats
            .groupby(['game_date', 'team'], observed=True)[value_cols]
            .mean()
            .rename(columns=lambda c: f"pen_{c}")
            .reset_index()
        )

        all_matchups_with_stats = all_matchups_with_stats.merge(usage_df, on=['game_date','team'], how='left')
        all_matchups_with_stats = all_matchups_with_stats.merge(bullpen_agg, on=['game_date','team'], how='left')

        all_matchups_with_stats = self._fill_bullpen_with_priors(all_matchups_with_stats, rl_priors)

        for col in ['last_app_date_team_starter', 'last_app_date_opposing_starter']:
            all_matchups_with_stats[col] = all_matchups_with_stats[col].fillna(pd.to_datetime(f"{self.season}-01-01"))
            
        logger.debug(f"Pitching Rosters with stats: {len(all_matchups_with_stats)}")
        logger.info(f" Rosters with stats rows with NAN\n{all_matchups_with_stats[all_matchups_with_stats.isna().any(axis=1)]}")
        logger.debug(f" all_matchups_with_stats_cols\n{all_matchups_with_stats.columns.to_list()}\n")

        first_cols = ['game_date', 'dh', 'season', 'team', 'opposing_team', 'name_team_starter', 
                      'normalized_player_name_team_starter', 'name_opposing_starter', 
                      'normalized_player_name_opposing_starter']
        
        remaining_cols = [col for col in all_matchups_with_stats.columns if col not in first_cols]

        all_matchups_with_stats = all_matchups_with_stats[first_cols + remaining_cols]

        logger.info(f" Final Pitcher Features\n{all_matchups_with_stats.head(5)}")

        with open("pitching_features.txt", "w") as f:
            f.write(all_matchups_with_stats.to_string())

        try:
            all_matchups_with_stats.to_parquet(cache_path, index=True)
            logger.info(f" Successfully cached pitching rolling stats to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache pitching rolling stats: {e}")

        return all_matchups_with_stats

    def _fill_bullpen_with_priors(self, df: DataFrame, priors):
        df = df.copy()
        for prior_name, val in priors.items():
            
            stat_name = f"pen_{prior_name[6:]}"
            prior_cols = [col for col in df.columns if stat_name in col]
            
            if prior_cols:
                for col in prior_cols:
                    df[col] = df[col].fillna(val)

        return df

    def _fill_missing_rosters(self, pitching_rosters: DataFrame, starters_map: DataFrame) -> DataFrame:
        """Fill missing roster data by forward-filling the most recent roster for each team"""
        logger.info(" Filling missing roster data...")
        
        rosters_plus = pitching_rosters.merge(starters_map, on=['game_date', 'team'], how='left')
        
        missing_entries = starters_map.merge(
            pitching_rosters[['game_date', 'team']].drop_duplicates(), 
            on=['game_date', 'team'], 
            how='left', 
            indicator=True
        )

        missing_entries = missing_entries[missing_entries['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        if len(missing_entries) > 0:
            logger.warning(f"Found {len(missing_entries)} missing roster entries, forward-filling...")
            
            filled_rosters = []
            
            for _, missing_row in missing_entries.iterrows():
                team = missing_row['team']
                game_date = missing_row['game_date']
                starter_id = missing_row['team_starter_id']
                
                team_rosters = pitching_rosters[pitching_rosters['team'] == team]
                recent_rosters = team_rosters[team_rosters['game_date'] < game_date]
                
                if len(recent_rosters) > 0:
                    most_recent_date = recent_rosters['game_date'].max()
                    most_recent_roster = recent_rosters[recent_rosters['game_date'] == most_recent_date].copy()
                    
                    most_recent_roster['game_date'] = game_date
                    most_recent_roster['team_starter_id'] = starter_id
                    
                    filled_rosters.append(most_recent_roster)
                    
                    logger.debug(f"Filled roster for {team} on {game_date} using roster from {most_recent_date}")
                else:
                    logger.warning(f"No previous roster data found for {team} before {game_date}")
            
            if filled_rosters:
                filled_df = pd.concat(filled_rosters, ignore_index=True)
                rosters_plus = pd.concat([rosters_plus, filled_df], ignore_index=True)
        
        return rosters_plus


    def _compute_bullpen_usage(self, bullpen_stats: DataFrame) -> DataFrame:
        logger.info(" Compute bullpen usage features...")
        df = bullpen_stats.copy()

        rest_days = (df['game_date'] - df['last_app_date']).dt.days
        rest_days = rest_days.fillna(4)
        df['rest_days'] = rest_days 
        df['freshness'] = (rest_days.clip(lower=0, upper=3) / 3.0)

        w = df.get('gmli_ewm_h8')
        if w is None:
            w = pd.Series(1.0, index=df.index)

        w = w.fillna(0).clip(lower=0)
        df['w_norm'] = w.groupby([df['game_date'], df['team']], observed=True).transform(
            lambda x: x / x.sum() if x.sum() > 0 else pd.Series(1.0 / len(x), index=x.index)
        )

        g = df.groupby(['game_date', 'team'], observed=True)
        out = pd.DataFrame({
            'pen_rest_days_mean'     : g['rest_days'].mean(),
            'pen_rest_days_median'   : g['rest_days'].median(),
            'pen_freshness_mean'     : g['freshness'].mean(),
            'pen_freshness_gmliw'    : (g.apply(lambda x: (x['freshness'] * x['w_norm']).sum())),
            'pen_hi_lev_available'   : g.apply(lambda x: ( (x['rest_days'] >= 1).astype(int) * x['w_norm'] ).sum()),
        }).reset_index()

        return out

    def _classify_pitcher_type(self) -> DataFrame:
        """Classifies each pitcher as a starter or reliever"""
        logger.info(" Classifying starter/reliever...")
        data = self.data.copy()
        lineups_data = self.pitching_matchups.copy()
        
        logger.info(f"Pitching Matchup NA rows\n{lineups_data[lineups_data.isna().any(axis=1)]}\n")
        
        merged_df = pd.merge(
            lineups_data,
            data,
            how='left',
            on=['game_date', 'team', 'dh'],
            suffixes=('', '_stats')
        )
        merged_df['team_starter_id'] = merged_df['team_starter_id'].astype("Int64")
        merged_df['player_id'] = merged_df['player_id'].astype("Int64")

        merged_df = merged_df.drop(columns=['season_stats'])

        cond = merged_df['team_starter_id'].eq(merged_df['player_id']).fillna(False)
        merged_df['is_starter'] = cond.astype(bool)

        return merged_df
        
    def _is_opener(self, pitcher_data: DataFrame) -> DataFrame:
        """Encodes the probable starter as a traditional starter or an opener in a bullpen game."""
        logger.info(" Classifying openers...")
        if 'is_starter' not in pitcher_data.columns:
            raise RuntimeError("This method is meant to be called after the method _is_starter() is called.")

        starter_stats = pitcher_data.copy()
        
        g = starter_stats.groupby('player_id', observed=True)
        games = g['games'].sum(min_count=1)
        games_started = g['gs'].sum(min_count=1)

        ratio = games_started.div(games).replace([np.inf, -np.inf], np.nan)
        opener_flag = (ratio.lt(0.25)).astype('Int64').fillna(0)

        opener_mapping = opener_flag.rename('is_opener_flag').reset_index()
        
        pitcher_data = pitcher_data.merge(
            opener_mapping, on='player_id', how='left'
        )

        pitcher_data['is_opener'] = np.where(
            pitcher_data['is_starter'] == True,
            pitcher_data['is_opener_flag'].fillna(False),
            False
        ).astype(bool)

        pitcher_data.drop(columns=['is_opener_flag'], inplace=True)

        return pitcher_data
    
    def _rolling_pitching_stats(self, pitcher_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Orchestrates computation of rolling stats for starters and relievers"""  

        relievers = pitcher_data[pitcher_data['is_starter'] == False].copy()
        starters = pitcher_data[pitcher_data['is_starter'] == True].copy()

        rl_df, rl_priors = self._compute_rolling_bullpen_stats(relievers)
        st_df, st_priors = self._compute_rolling_starter_stats(starters)
        
        st_df_velo = self._compute_starter_velo_trends(st_df)

        return rl_df, st_df_velo, rl_priors, st_priors
        
    def _compute_rolling_bullpen_stats(self, df: DataFrame, halflives=(3, 8, 20)) -> DataFrame:
        "Computes the rolling stats of the bullpen"
        logger.info(" Calculating rolling bullpen stats...")

        rel = df.copy()
        rel.sort_values(['player_id', 'game_date', 'dh'], inplace=True)

        rel['apps'] = 1
        rel["wpa_li"] = rel["wpa"] / rel["gmli"].replace(0, np.nan)

        prior_specs = {
            "prior_k_percent"     : ("k_percent",      "tbf"),
            "prior_bb_percent"    : ("bb_percent",     "tbf"),
            "prior_babip"         : ("babip",          "bip"),
            "prior_barrel_percent": ("barrel_percent", "bip"),
            "prior_hard_hit"      : ("hard_hit",       "bip"),
            "prior_ev"            : ("ev",             "bip"),
            "prior_hr_fb"         : ("hr_fb",          "bip"),
            "prior_fip"           : ("fip",            "ip"),
            "prior_siera"         : ("siera",          "ip"),
            "prior_stuff"         : ("stuff",          "tbf"),
            "prior_gmli"          : ("gmli",           "apps"),
            "prior_wpa_li"        : ("wpa_li",         "apps")
        }

        shrinkage_weights_cols = ['tbf', 'bip', 'ip', 'apps']

        specs = {
            'k_percent'     : ('k_percent',      'tbf',  'prior_k_percent',       120, True),
            'bb_percent'    : ('bb_percent',     'tbf',  'prior_bb_percent',      120, True),
            'babip'         : ('babip',          'bip',  'prior_babip',           180, True),
            'barrel_percent': ('barrel_percent', 'bip',  'prior_barrel_percent',  100, True),
            'hard_hit'      : ('hard_hit',       'bip',  'prior_hard_hit',        100, True),
            'ev'            : ('ev',             'bip',  'prior_ev',              180, True),
            'hr_fb'         : ('hr_fb',          'bip',  'prior_hr_fb',            80, True),
            'fip'           : ('fip',            'ip',   'prior_fip',              20, True),
            'siera'         : ('siera',          'ip',   'prior_siera',            20, True),
            'stuff'         : ('stuff',          'tbf',  'prior_stuff',            40, True),
            'gmli'          : ('gmli',           'apps', 'prior_gmli',             10, True),
            'wpa_li'        : ('wpa_li',         'apps', 'prior_wpa_li',           10, True),
        }

        cols = ['player_id', 'mlb_id', 'game_date','dh','team','season','name','normalized_player_name', 
                'is_starter', 'is_opener']

        result, reliever_priors = BaseFeatures.compute_rolling_stats(
            data=rel,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=specs,
            preserve_cols=cols,
            halflives=halflives
        )
        
        return result, reliever_priors
        
    def _compute_rolling_starter_stats(self, df: DataFrame, halflives=(3, 8, 20)) -> DataFrame:
        logger.info(" Calculating rolling starter stats...")    

        start = df.copy()
        start.sort_values(['player_id', 'game_date', 'dh'], inplace=True)
        start['apps'] = 1

        prior_specs = {
            "prior_k_percent"     : ("k_percent",      "tbf"),
            "prior_bb_percent"    : ("bb_percent",     "tbf"),
            "prior_babip"         : ("babip",          "bip"),
            "prior_barrel_percent": ("barrel_percent", "bip"),
            "prior_hard_hit"      : ("hard_hit",       "bip"),
            "prior_ev"            : ("ev",             "bip"),
            "prior_hr_fb"         : ("hr_fb",          "bip"),  
            "prior_fip"           : ("fip",            "ip"),
            "prior_siera"         : ("siera",          "ip"),
            "prior_stuff"         : ("stuff",          "tbf"),
            "prior_wpa"           : ("wpa",         "apps")
        }

        shrinkage_weights_cols = ['tbf', 'bip', 'ip', 'apps']
        
        specs = {
            'k_percent'     : ('k_percent',      'tbf',  'prior_k_percent',       100, True),
            'bb_percent'    : ('bb_percent',     'tbf',  'prior_bb_percent',      100, True),
            'babip'         : ('babip',          'bip',  'prior_babip',           150, True),
            'barrel_percent': ('barrel_percent', 'bip',  'prior_barrel_percent',  120, True),
            'hard_hit'      : ('hard_hit',       'bip',  'prior_hard_hit',        120, True),
            'ev'            : ('ev',             'bip',  'prior_ev',              120, True),
            'hr_fb'         : ('hr_fb',          'bip',  'prior_hr_fb',            80, True),
            'fip'           : ('fip',            'ip',   'prior_fip',              50, True),
            'siera'         : ('siera',          'ip',   'prior_siera',            50, True),
            'stuff'         : ('stuff',          'tbf',  'prior_stuff',            40, True),
            'wpa'           : ('wpa',         'apps', 'prior_wpa',              10, True),
        }
        
        cols = ['player_id', 'mlb_id', 'game_date','dh','team','season','name',
                'normalized_player_name', 'is_starter', 'is_opener',
                'fa_percent', 'fc_percent', 'si_percent', 'fa_velo',
                'fc_velo', 'si_velo']

        result, starter_priors = BaseFeatures.compute_rolling_stats(
            data=start,
            prior_specs=prior_specs,
            shrinkage_weights_cols=shrinkage_weights_cols,
            ewm_cols=specs,
            preserve_cols=cols,
            halflives=halflives)
        
        return result, starter_priors
        

    def _compute_starter_velo_trends(self, starters: DataFrame) -> DataFrame:
        """Computes the ewm velocity difference of fastballs, sinkers, and cutters
        for a starter, weighted by pitch usage"""

        logger.info(" Computing starter velo trends...")

        df = starters.copy()
        df = df.sort_values(['player_id', 'game_date'])
        df['gs'] = 1

        for col in ['fa_percent', 'fc_percent', 'si_percent']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        by = df['player_id']
        
        velo_weighted_sum = (df['fa_velo'] * df['fa_percent']) + \
                           (df['fc_velo'] * df['fc_percent']) + \
                           (df['si_velo'] * df['si_percent'])
        
        usage_agg = df['fa_percent']  + df['fc_percent']  + df['si_percent']
        

        velo_season, _ = BaseFeatures.expanding_weighted_mean(velo_weighted_sum, usage_agg, by, val_is_rate=False)
        ewm_short = BaseFeatures.compute_ewm(velo_weighted_sum, usage_agg, by, half_life=(2), val_is_rate=False)
        ewm_long = BaseFeatures.compute_ewm(velo_weighted_sum, usage_agg, by, half_life=(6), val_is_rate=False)

        df["velo_season"] = velo_season
        df["velo_short"] = ewm_short
        df["velo_long"] = ewm_long
        df["velo_diff"] = ewm_short - ewm_long

        velo_avg = velo_weighted_sum / usage_agg.replace(0, np.nan)
        valid_velo_mask = ~velo_avg.isna() & (velo_avg > 0)
        if valid_velo_mask.sum() > 0:
            avg_starter_velo = velo_avg[valid_velo_mask].mean()
            logger.info(f" Average starter velocity for filling NaN values: {avg_starter_velo:.2f} mph")
        else:
            avg_starter_velo = 91.0
            logger.warning(f" No valid velocity data found, using default {avg_starter_velo} mph")

        for col in ['velo_season', 'velo_short', 'velo_long']:
            df[col] = df[col].fillna(avg_starter_velo)
        
        df["velo_diff"] = df["velo_diff"].fillna(0)

        df.drop(columns=['fa_velo', 'fa_percent', 'fc_velo', 
                         'fc_percent', 'si_velo', 'si_percent' , 'gs'], inplace=True)
        return df

    

def main():
    from src.data.loaders.player_loader import PlayerLoader
    loader = PlayerLoader()
    pitching_data = loader.load_for_season_pitcher(2021)

    feats = PitchingFeatures(2021, pitching_data)
    _ = feats.load_features()
    # pitcher_data = feats._classify_pitcher_type()
    # pitcher_data = feats._is_opener(pitcher_data)
    # p_feats = feats._rolling_pitching_stats(pitcher_data)
if __name__ == "__main__":
    main()
    
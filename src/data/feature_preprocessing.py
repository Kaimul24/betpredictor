from src.data.features.feature_pipeline import FeaturePipeline
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import logging, sys, argparse, os
from typing import List, Tuple, Dict, Union
from sklearn.preprocessing import StandardScaler
from src.config import PROJECT_ROOT, FEATURES_CACHE_PATH
import pickle
import joblib

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()



LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_preprocessing.log"

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Feature preprocessing runner")
    parser.add_argument("--force-recreate", action="store_true", help="Recreate rolling features, even if cached file exists")
    parser.add_argument("--force-recreate-preprocessing", action="store_true", help="Recreate preprocessed datasets, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    return parser.parse_args()


def setup_logging(args=None):
    """Configure logging based on CLI arguments"""
    logger = logging.getLogger("feature_preprocessing")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False 

    if logger.handlers:
        logger.handlers.clear()
    
    fmt = logging.Formatter(
        "%(levelname)s:%(name)s:%(message)s"
    )
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    

    if args and hasattr(args, 'log') and args.log:
        log_file = args.log_file if hasattr(args, 'log_file') and args.log_file else LOG_FILE
        
        if hasattr(args, 'clear_log') and args.clear_log:
            try:
                with open(log_file, 'w') as f:
                    pass
                logger.info(f" Cleared log file: {log_file}")
            except Exception as e:
                logger.info(f" Warning: Could not clear log file {log_file}: {e}")
        
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f" Logging to file: {log_file}")
    
    return logger



class PreProcessing():

    def __init__(self, seasons: List[int]):
        self.seasons = seasons
        self.seasons_str = "_".join(map(str, seasons))

        
        # Define cache paths for each dataset
        self.cache_dir = FEATURES_CACHE_PATH / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_paths = {
            'X_train': self.cache_dir / f"X_train_seasons_{self.seasons_str}.parquet",
            'y_train': self.cache_dir / f"y_train_seasons_{self.seasons_str}.parquet", 
            'X_val': self.cache_dir / f"X_val_seasons_{self.seasons_str}.parquet",
            'y_val': self.cache_dir / f"y_val_seasons_{self.seasons_str}.parquet",
            'X_test': self.cache_dir / f"X_test_seasons_{self.seasons_str}.parquet",
            'y_test': self.cache_dir / f"y_test_seasons_{self.seasons_str}.parquet",
            'odds_data': self.cache_dir / f"odds_data_seasons_{self.seasons_str}.parquet",
            'scaler': self.cache_dir / f"scaler_seasons_{self.seasons_str}.pkl"
        }

        self.target = ['is_winner_home']

        self.exclude_columns = [
            'home_team_score', 'away_team_score', 'home_score', 
            'winning_team', 'losing_team', 'winner', 'away_score',
            
            'game_datetime', 'season', 'is_home',
            
            'away_starter_normalized', 'home_starter_normalized',
            'starter_normalized', 'home_opposing_starter_normalized', 'home_opposing_starter_normalized', 'home_team_starter_normalized_player_name',
            'away_team_starter_normalized_player_name', 'home_team_starter_name', 'away_team_starter_name', 'away_opposing_starter_normalized',
            
            'home_team_starter_id', 'away_starter_id','away_team_starter_id', 'away_team_starter_player_id', 'mlb_id', 'venue_id',

            'status', 'venue_name', 'venue_timezone', 'venue_gametime_offset'
        ]

    def _remove_early_games(self, dfs: List[DataFrame]) -> List:
        removed_early = []
        for df in dfs:  # This tries to iterate over a DataFrame
            print(df.columns.to_list())
            d = df[(df['home_team_gp'] > 10) & (df['away_team_gp'] > 10)]
            removed_early.append(d)
        return removed_early
           
    def _feature_scaling(self, dfs: List[DataFrame], is_xgboost: bool = False) -> Dict[str, Union[DataFrame, StandardScaler | None]]:

        filtered_dfs = [df[[col for col in df.columns if col not in self.exclude_columns]] for df in dfs]
        
        train_dfs = filtered_dfs[:2]
        val_df = filtered_dfs[-2].reset_index()
        test_df = filtered_dfs[-1].reset_index()

        val_df = val_df.set_index(['season', 'game_date', 'dh', 'home_team', 'away_team', 'game_id']).sort_index()
        test_df = test_df.set_index(['season', 'game_date', 'dh', 'home_team', 'away_team', 'game_id']).sort_index()

        train_data = pd.concat(train_dfs)

        self.logger.info(f" Dropping {train_data.isna().any(axis=1).sum()} rows from training data...")
        train_data = train_data.dropna()

        self.logger.info(f" Dropping {val_df.isna().any(axis=1).sum()} rows from validation data...")
        val_df = val_df.dropna()

        self.logger.info(f" Dropping {test_df.isna().any(axis=1).sum()} rows from test data...")
        test_df = test_df.dropna()

        self.logger.debug(f" Dtypes\n{train_dfs[0].dtypes.value_counts()}")
        self.logger.debug(f" Features\n{train_dfs[0].columns.to_list()}")
        self._log_nan_rows(train_dfs)

        X_train = train_data.drop(columns=self.target)
        y_train = train_data[self.target]

        X_val = val_df.drop(columns=self.target)
        y_val = val_df[self.target]

        X_test = test_df.drop(columns=self.target)
        y_test = test_df[self.target]

        bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
        numeric_cols = X_train.select_dtypes(exclude=['bool']).columns.tolist()
        object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        self.logger.info(f" Boolean columns: {bool_cols}")
        self.logger.info(f" Object cols: {object_cols}")
        scaler = None

        if not is_xgboost:
            self.logger.info(f" Numeric columns to scale: {len(numeric_cols)}")
            scaler = StandardScaler()

            if bool_cols:
                X_train_numeric_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train[numeric_cols]),
                    columns=numeric_cols,
                    index=X_train.index
                )
                
                X_train_bool = X_train[bool_cols].astype('int')
                
                X_train = pd.concat([X_train_numeric_scaled, X_train_bool], axis=1)
                
                X_val_numeric_scaled = pd.DataFrame(
                    scaler.transform(X_val[numeric_cols]),
                    columns=numeric_cols,
                    index=X_val.index
                )
                X_val_bool = X_val[bool_cols].astype('int')
                X_val = pd.concat([X_val_numeric_scaled, X_val_bool], axis=1)
                
                X_test_numeric_scaled = pd.DataFrame(
                    scaler.transform(X_test[numeric_cols]),
                    columns=numeric_cols,
                    index=X_test.index
                )
                X_test_bool = X_test[bool_cols].astype('int')
                X_test = pd.concat([X_test_numeric_scaled, X_test_bool], axis=1)

        self.logger.debug(f" Dtypes\n{X_train.dtypes.value_counts()}")
        self.logger.debug(f" Final X_train head\n{X_train.head(3).to_string()}")
        self._log_nan_rows([X_train, X_val, X_test])
        
        self.scaler = scaler
            
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler
        }

    def _separate_odds_cols(self, dfs: List[DataFrame]) -> List[Tuple[DataFrame, DataFrame]]:
        results = []

        for df in dfs:
            odds_cols = [col for col in df.columns if 'odds' in col]
            
            feat_df = df[[col for col in df.columns if col not in odds_cols]]
            odds_df = df[odds_cols] if odds_cols else DataFrame(index=df.index)
            
            results.append((feat_df, odds_df))
        
        return results
        
    def _get_features(self, force_recreate: bool = False, clear_log: bool = False) -> Tuple[List[DataFrame], List[DataFrame]]:
        all_features = []
        all_odds = []

        for year in self.seasons:
            feat_pipe = FeaturePipeline(year, logger=self.logger)
            season_feats, odds_data = feat_pipe.start_pipeline(force_recreate, clear_log)
            all_features.append(season_feats)
            all_odds.append(odds_data)
        
        return all_features, all_odds
    
    def _log_nan_rows(self, dfs: List[DataFrame]) -> None:
        for i, df in enumerate(dfs):
            na_rows = df[df.isna().any(axis=1)]
            if not na_rows.empty:
                self.logger.debug(f" --- DataFrame {i+1} ---")
                for idx, row in na_rows.iterrows():
                    nan_cols = row[row.isna()].index.tolist()
                    self.logger.debug(f" Row {idx}: NaN columns = {nan_cols}")
                self.logger.debug(f"\n{na_rows}")
            else:
                self.logger.debug(f" --- DataFrame {i+1} --- No NaN rows found")
    
    def _save_cached_data(self, processed_data: Dict, odds_data: DataFrame) -> None:
        """Save processed data to cache files"""
        try:
            for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
                processed_data[key].to_parquet(self.cache_paths[key], index=True)
                self.logger.info(f" Cached {key} to {self.cache_paths[key]}")
            
            odds_data.to_parquet(self.cache_paths['odds_data'], index=True)
            self.logger.info(f" Cached odds to {self.cache_paths['odds_data']}")

            joblib.dump(processed_data['scaler'], self.cache_paths['scaler'])
            self.logger.info(f" Cached scaler to {self.cache_paths['scaler']}")
            
        except Exception as e:
            self.logger.error(f" Failed to cache processed data: {e}")
    
    def _load_cached_data(self) -> Union[Tuple[Dict, DataFrame], None]:
        """Load processed data from cache files"""
        try:
            cached_data = {}
            
            for key in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
                cached_data[key] = pd.read_parquet(self.cache_paths[key])
                self.logger.info(f" Loaded cached {key}")
    

            odds_df = pd.read_parquet(self.cache_paths['odds_data'])
            self.logger.info(" Loaded cached odds")

            cached_data['scaler'] = joblib.load(self.cache_paths['scaler'])
            self.logger.info(f" Loaded cached scaler")
            
            return cached_data, odds_df
            
        except Exception as e:
            self.logger.error(f" Failed to load cached data: {e}")
            return None
    
    def _cache_exists(self) -> bool:
        """Check if all required cache files exist"""
        required_files = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'scaler']
        return all(self.cache_paths[key].exists() for key in required_files)
    
    def _clear_cache(self) -> None:
        """Remove all cached files"""
        for cache_path in self.cache_paths.values():
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    self.logger.info(f" Removed cached file: {cache_path}")
                except Exception as e:
                    self.logger.error(f" Error removing cache file {cache_path}: {e}")
    
    def preprocess_feats(self, force_recreate: bool = False, force_recreate_preprocessing: bool = False, clear_log: bool = False, is_xgboost: bool = False) -> Tuple[Dict, DataFrame]:
        """
        Main preprocessing method with caching functionality.
        
        Args:
            force_recreate: If True, recreate underlying rolling features even if cached
            force_recreate_preprocessing: If True, recreate preprocessed datasets even if cached
            clear_log: If True, clear the log file before starting
            
        Returns:
            Dictionary containing processed features and data splits
        """
        if not hasattr(self, 'logger'):
            self.logger = setup_logging()

        if not force_recreate_preprocessing and self._cache_exists():
            self.logger.info(f" Found cached preprocessed data for seasons {self.seasons}")
            cached_data = self._load_cached_data()
            if cached_data is not None:
                self.logger.info(f" Successfully loaded cached preprocessed data")
                return cached_data[0], cached_data[1]
            else:
                self.logger.warning(f" Failed to load cached data, reprocessing...")
        
        elif force_recreate_preprocessing:
            self.logger.info(f" Force recreate preprocessing enabled, clearing cache and reprocessing...")
            self._clear_cache()
        
        else:
            self.logger.info(f" No cached preprocessed data found, processing features...")

        self.logger.info(f" Getting raw features for seasons {self.seasons}")
        features, odds_data = self._get_features(force_recreate, clear_log)

        self.logger.info(f" Performing feature scaling and data splitting")
        processed_data = self._feature_scaling(features, is_xgboost)

        all_odds = pd.concat(odds_data)
        
        self.logger.info(f" Caching processed data")
        self._save_cached_data(processed_data, all_odds)
        
        return processed_data, all_odds


def main():
    args = create_args()  # Get args from return value
    
    # Set up logging first, before creating PreProcessing instance
    logger = setup_logging(args)
    
    pre_processor = PreProcessing([2021, 2022, 2023])
    pre_processor.logger = logger  # Assign the logger
    
    preprocessed_feats, odds_data = pre_processor.preprocess_feats(
        force_recreate=args.force_recreate,
        force_recreate_preprocessing=args.force_recreate_preprocessing,
        clear_log=args.clear_log
    )

    print(preprocessed_feats.keys())


if __name__ == "__main__":
    main()
from src.data.features.feature_pipeline import FeaturePipeline

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import logging, sys, argparse
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from config import PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_preprocessing.log"

def create_args():
    """Parse command line arguments"""
    global args
    parser = argparse.ArgumentParser(description="Feature preprocessing runner")
    parser.add_argument("--force-recreate", action="store_true", help="Recreate batting rolling features, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    args = parser.parse_args()

def setup_logging():
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
    

    if hasattr(args, 'log') and args.log:
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
        self.logger = setup_logging()
        self.seasons = seasons

        self.target = ['is_winner']

        self.exclude_columns = [
            # Game outcome information (data leakage)
            'team_score', 'opposing_team_score', 'home_score', 
            'winning_team', 'losing_team',
            
            # Identifiers and metadata
            'game_id', 'game_datetime', 'season',
            
            # Player names (categorical with too many unique values)
            'away_starter_normalized', 'home_starter_normalized',
            'starter_normalized', 'opposing_starter_normalized',
            'name_team_starter', 'normalized_player_name_team_starter',
            'name_opposing_starter', 'normalized_player_name_opposing_starter',
            
            # IDs and foreign keys
            'team_starter_id', 'opposing_starter_id', 'player_id_team_starter',
            'player_id_opposing_starter', 'mlb_id', 'venue_id',
            
            # Date columns (temporal leakage risk)
            'last_app_date_team_starter', 'last_app_date_opposing_starter',
            
            # Status and other metadata
            'status', 'venue_name', 'venue_timezone', 'venue_gametime_offset'
        ]

    def _feature_scaling(self, dfs: List[Tuple[DataFrame, DataFrame]]):

        stat_dfs = [df[0] for df in dfs]
        odds_dfs = [df[1] for df in dfs]
        
        filtered_dfs = [df[[col for col in df.columns if col not in self.exclude_columns]] for df in stat_dfs]
        
        train_dfs = filtered_dfs[:1]
        test_val_dfs = filtered_dfs[-1].reset_index()
        
        val_test_split_date = pd.to_datetime('2024-6-30')

        val_data = test_val_dfs.loc[test_val_dfs['game_date'] <= val_test_split_date]
        test_data = test_val_dfs.loc[test_val_dfs['game_date'] > val_test_split_date]
        
        val_data = val_data.set_index(['season', 'game_date', 'dh', 'team', 'opposing_team'])
        test_data = test_data.set_index(['season', 'game_date', 'dh', 'team', 'opposing_team'])

        train_data = pd.concat(train_dfs)

        self.logger.info(f" val df\n{val_data.tail(3)}")
        self.logger.info(f" test df\n{test_data.head(3)}")

        self.logger.debug(f" Dtypes\n{train_dfs[0].dtypes.value_counts()}")
        self._log_nan_rows(train_dfs)

        X_train = train_data.drop(columns=self.target)
        y_train = train_data[self.target]

        self.logger.debug(f" X_train\n{X_train.to_string()}")

        X_val = val_data.drop(columns=self.target)
        y_val = val_data[self.target]

        X_test = test_data.drop(columns=self.target)
        y_test = test_data[self.target]

        bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
        numeric_cols = X_train.select_dtypes(exclude=['bool']).columns.tolist()
        
        self.logger.info(f" Boolean columns: {bool_cols}")
        self.logger.info(f" Numeric columns to scale: {len(numeric_cols)}")

        scaler = StandardScaler()

        if bool_cols:
            X_train_numeric_scaled = pd.DataFrame(
                scaler.fit_transform(X_train[numeric_cols]),
                columns=numeric_cols,
                index=X_train.index
            )
            
            X_train_bool = X_train[bool_cols].astype('int8')
            
            X_train_scaled = pd.concat([X_train_numeric_scaled, X_train_bool], axis=1)
            
            X_val_numeric_scaled = pd.DataFrame(
                scaler.transform(X_val[numeric_cols]),
                columns=numeric_cols,
                index=X_val.index
            )
            X_val_bool = X_val[bool_cols].astype('int8')
            X_val_scaled = pd.concat([X_val_numeric_scaled, X_val_bool], axis=1)
            
            X_test_numeric_scaled = pd.DataFrame(
                scaler.transform(X_test[numeric_cols]),
                columns=numeric_cols,
                index=X_test.index
            )
            X_test_bool = X_test[bool_cols].astype('int8')
            X_test_scaled = pd.concat([X_test_numeric_scaled, X_test_bool], axis=1)

        self.logger.debug(f" Dtypes\n{X_train_scaled.dtypes.value_counts()}")
        self._log_nan_rows([X_train_scaled, X_val_scaled, X_test_scaled])
        
        self.scaler = scaler
            
        return {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'odds_dfs': odds_dfs,
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
        
    def _get_features(self) -> List[DataFrame]:
        all_features = []

        for year in self.seasons:
            feat_pipe = FeaturePipeline(year)
            season_feats = feat_pipe.start_pipeline(args.force_recreate, args.clear_log)
            all_features.append(season_feats)
        
        return all_features
    
    def _log_nan_rows(self, dfs: List[DataFrame]) -> None:
        for i, df in enumerate(dfs):
            na_rows = df[df.isna().any(axis=1)]
            self.logger.debug(f" --- DataFrame {i} ---\n{na_rows}")
    
    def preprocess_feats(self, force_recreate: bool = False, clear_log: bool = False) -> Dict:
        global args
        if args is None:
            parser = argparse.ArgumentParser(description="Feature engineering runner")
            parser.add_argument("--force-recreate", action="store_true", help="Recreate batting rolling features, even if cached file exists")
            parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
            parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
            parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
            args = parser.parse_args([])
            args.force_recreate = force_recreate
            args.clear_log = clear_log

        raw_feats = self._get_features()
        odds_separated_dfs = self._separate_odds_cols(raw_feats)

        processed_data = self._feature_scaling(odds_separated_dfs)
        
        return processed_data


def main():
    create_args()
    pre_processor = PreProcessing([2021, 2022, 2023, 2024])
    preprocessed_feats = pre_processor.preprocess_feats()
    #print(preprocessed_feats)

if __name__ == "__main__":
    main()
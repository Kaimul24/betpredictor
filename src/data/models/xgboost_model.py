from pandas.core.api import DataFrame as DataFrame
import pandas as pd
import logging, sys, argparse, json, os
import xgboost as xgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict
from tqdm import tqdm

from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "xgboost_model_log.log"

HYPERPARAM_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_hyperparameters"
HYPERPARAM_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_models"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = PROJECT_ROOT / "src" / "data" / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CAL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "calibrators"
CAL_DIR.mkdir(parents=True, exist_ok=True)

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XGBoost Model Pipeline")
    parser.add_argument("--retune", action='store_true', help='Retune hyperparameters')
    parser.add_argument("--force-recreate", action="store_true", help="Recreate rolling features, even if cached file exists")
    parser.add_argument("--force-recreate-preprocessing", action="store_true", help="Recreate preprocessed datasets, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    return parser.parse_args()

def setup_logging(args):
    """Configure logging based on CLI arguments"""
    logger = logging.getLogger("xgboost_model")
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

class XGBoostModel:

    def __init__(self, model_args, logger, all_data: Dict):
        self.model_args = model_args
        self.logger = logger

        self.hyperparam_path = HYPERPARAM_DIR / "xgboost_hyperparams.json"       

        if all_data == {}:
            raise ValueError(f" No data was supplied.")

        X_train = all_data['X_train']
        y_train = all_data['y_train']
        X_val = all_data['X_val']
        y_val = all_data['y_val']
        X_test = all_data['X_test']
        y_test = all_data['y_test']

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test

        self.dtrain = dtrain
        self.dtest = dtest

        n_val = len(self.y_val)
        cutoff = int(0.5 * n_val)
        self.dval_es = xgb.DMatrix(self.X_val[:cutoff], label=self.y_val[:cutoff])
        self.dcal    = xgb.DMatrix(self.X_val[cutoff:], label=self.y_val[cutoff:])
        self.y_cal   = self.y_val[cutoff:]
        

    def train_and_eval_model(self, retune_hyperparams: bool = False):
        """
        Train and evaluate the XGBoost model.
        
        Args:
            force_recreate: If True, recreate underlying rolling features even if cached
            force_recreate_preprocessing: If True, recreate preprocessed datasets even if cached
            clear_log: If True, clear the log file before starting
        """
        
        retune_hyperparams = self.model_args.retune

        if retune_hyperparams or not self.hyperparam_path.exists():
            self.logger.info(f" Performing hyperparameter tuning")
            best_params = self._hyperparam_tune()
            self._save_hyperparams(best_params)
        else:
            self.logger.info(" Loading existing hyperparameters")
            best_params = self._load_hyperparameters()

        trained_model = self.train(best_params)

        return trained_model

    def _hyperparam_tune(self):
        pbar = tqdm(total=200, desc="Optuna HPO")

        def tqdm_callback(study, trial):
            pbar.update(1)

        def objective(trial):
            
            params = {
                    'verbosity': 1,
                    'objective': 'binary:logitraw',
                    'device': "cpu",
                    'tree_method': 'hist',
                    'eval_metric': 'logloss',
                    "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                    'seed': 42,
                    'nthread': 12, 
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'grow_policy':  trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                    
                }
            
            if params['booster'] == 'dart':
                params['normalize_type'] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                params['sample_type'] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                params['rate_drop'] = trial.suggest_float('rate_drop', 1e-5, 1.0)
                params['skip_drop']= trial.suggest_float('skip_drop', 1e-5, 1.0)
            
            dtrain = self.dtrain
            tscv = TimeSeriesSplit(n_splits=5)

            folds = list(tscv.split(np.arange(len(self.X_train))))
            
            scores = []

            for i, (tr_idx, val_idx) in enumerate(folds):
                dtr = dtrain.slice(tr_idx)
                dvl = dtrain.slice(val_idx)

                bst = xgb.train(
                    params,
                    dtr,
                    num_boost_round=2000,
                    evals=[(dtr, 'train'), (dvl, 'eval')],
                    early_stopping_rounds = 50,
                    verbose_eval=False
                )

                scores.append(bst.best_score)
                trial.report(sum(scores)/len(scores), step=i)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(sum(scores)/len(scores))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200, callbacks=[tqdm_callback])
        pbar.close()

        self.logger.info(f" Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        self.logger.info(f" Best Trial: {trial.value}")
        self.logger.info(f" Best params:  {trial.params.items()}")


        #self._vis_hpo(study)

        return study.best_params
    
    # def _vis_hpo(self, study):
    #     optuna.visualization.plot_optimization_history(study)
    #     optuna.visualization.plot_param_importances(study)

    def _plot_roc(self, y_true, y_proba, split: str):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({split})")
        plt.legend(loc="lower right")
        out_path = PLOTS_DIR / f"xgboost_roc_{split}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        self.logger.info(f" Saved ROC curve ({split}) to {out_path}")


    def train(self, hyperparams: Dict = None):

        BASE_PARAMS = {
                    'objective': 'binary:logitraw',
                    'device': "cpu",
                    'tree_method': 'hist',
                    'eval_metric': 'logloss',
                    'seed': 42,
                    'nthread': 12, 
            }
        
        all_params = {**BASE_PARAMS, **hyperparams}

        self.logger.info(" Training XGBoost model on CPU...")
        
        bst = xgb.train(
            all_params,
            self.dtrain,
            num_boost_round=1000,
            evals=[(self.dtrain, 'train'), (self.dval_es, 'eval')],
            verbose_eval=10,
            early_stopping_rounds=50
        )

        self.logger.info(f" Uncalibrated scores...")
        p_es = bst.predict(self.dval_es, iteration_range=(0, bst.best_iteration + 1))
        ll_es    = log_loss(self.y_val[:len(p_es)], p_es)
        auc_es   = roc_auc_score(self.y_val[:len(p_es)], p_es)
        brier_es = brier_score_loss(self.y_val[:len(p_es)], p_es)
        
        self.logger.info(f" Val(ES) Log Loss: {ll_es}")
        self.logger.info(f" Val(ES) ROC AUC: {auc_es}")
        self.logger.info(f" Val(ES) Brier:   {brier_es}")

        filepath = PLOTS_DIR / 'xgboost_calibration'
        out_path_curve = plot_calibration(self.y_val[:len(p_es)], p_es, split="val_es", filepath=filepath, n_bins=25)
        self.logger.info(f" Saved calibration curve (\"val_es\") to {out_path_curve}")

        self.logger.info(" Predictions on calibration set...")
        p_cal = bst.predict(self.dcal, iteration_range=(0, bst.best_iteration + 1))
        min_bin_size = 100
        calibrator, metrics = select_and_fit_calibrator(
            y_cal=self.y_cal,
            p_cal=p_cal,
            methods=("platt", "isotonic"),
            selection_metric="ece",
            min_bin_size=min_bin_size,
            n_bins=25,
            strategy="quantile",
        )
        ll_cal    = log_loss(self.y_val[len(p_es):], p_cal)
        auc_cal   = roc_auc_score(self.y_val[len(p_es):], p_cal)
        brier_cal = brier_score_loss(self.y_val[len(p_es):], p_cal)

        self.logger.info(f" Val(CAL) Log Loss: {ll_cal}")
        self.logger.info(f" Val(CAL) ROC AUC: {auc_cal}")
        self.logger.info(f" Val(CAL) Brier:   {brier_cal}")
        
        self.logger.info(f" Calibration candidates: {metrics}")
        save_calibrator(calibrator, str(CAL_DIR / "xgb_calibrator.json"))

        self.logger.info(" Evaluating calibrated predictions on calibration set...")
        p_cal_calibrated = apply_calibration(calibrator, p_cal)
        
        ll_cal_after = log_loss(self.y_cal, p_cal_calibrated)
        auc_cal_after = roc_auc_score(self.y_cal, p_cal_calibrated)
        brier_cal_after = brier_score_loss(self.y_cal, p_cal_calibrated)
        
        self.logger.info(f" Val(CAL) Post-Calibration Log Loss: {ll_cal_after}")
        self.logger.info(f" Val(CAL) Post-Calibration ROC AUC: {auc_cal_after}")
        self.logger.info(f" Val(CAL) Post-Calibration Brier:   {brier_cal_after}")
        
        # Calculate improvement metrics
        ll_improvement = ll_cal - ll_cal_after
        brier_improvement = brier_cal - brier_cal_after
        
        self.logger.info(f" Log Loss improvement: {ll_improvement:.4f}")
        self.logger.info(f" Brier Score improvement: {brier_improvement:.4f}")

        self._save_model(bst)

        self._plot_roc(self.y_cal, p_cal, split="cal")
        out_path_curve = plot_calibration(self.y_cal, p_cal, split="cal_after", filepath=filepath, n_bins=25, min_bin_size=min_bin_size)

        self.logger.info(f" Saved calibration curve (\"cal_after\") to {out_path_curve}")
        

        return bst
    

    def _save_hyperparams(self, params: Dict) -> None:
        self.hyperparam_path = HYPERPARAM_DIR / "xgboost_hyperparams.json"
        self.hyperparam_path.write_text(json.dumps(params, indent=4))

    def _load_hyperparameters(self) -> Dict:
        if self.hyperparam_path.exists():
            return json.loads(self.hyperparam_path.read_text())
        
        self.logger.warning(f"No hyperparameters found at {self.hyperparam_path}")
        return {}

    def predict(self, model: xgb.Booster = None):
        if not model:
            model = self._load_model()

        if model is None:
            self.logger.error(f" No model has been trained!")
            return
        
        self.logger.info(f" Predicting on test set")
        p_test_raw = model.predict(self.dtest, iteration_range=(0, model.best_iteration + 1))

        try:
            cal = load_calibrator(str(CAL_DIR / "xgb_calibrator.json"))
            p_test = apply_calibration(cal, p_test_raw)
            self.logger.info(" Applied calibrator to test predictions.")
        except FileNotFoundError:
            p_test = p_test_raw
            self.logger.info(" No calibrator found; using raw probabilities.")
        

        test_log_loss = log_loss(self.y_test, p_test)
        test_roc_auc  = roc_auc_score(self.y_test, p_test)
        test_brier    = brier_score_loss(self.y_test, p_test)

        self.logger.info(f" Test Log Loss: {test_log_loss}")
        self.logger.info(f" Test ROC AUC: {test_roc_auc}")
        self.logger.info(f" Test Brier Score: {test_brier}")

        self._plot_roc(self.y_test, p_test, split="test")

        filepath = PLOTS_DIR / 'xgboost_calibration'
        out_path_curve = plot_calibration(self.y_test, p_test, split="test", filepath=filepath, n_bins=25, min_bin_size=100)
        self.logger.info(f" Saved calibration curve (\"test\") to {out_path_curve}")

        return p_test
    
    def _save_model(self, model: xgb.Booster) -> None:
        xgboost_model_path = os.path.join(SAVED_MODEL_DIR, "saved_xgboost.json")
        self.xgboost_model_path = xgboost_model_path
        model.save_model(str(xgboost_model_path))

        self.logger.info(f" Succesfully saved XGBoost model to {xgboost_model_path}")

    
    def _load_model(self) -> xgb:
        model_path = SAVED_MODEL_DIR / "saved_xgboost.json"
        if model_path.exists():
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            self.logger.info(f" Succesfully loaded XGBoost model")
            return bst
        
        self.logger.warning(f" No XGBoost model found")
        return None

def main():
    model_args = create_args()
    logger = setup_logging(model_args)

    all_data = PreProcessing([2021, 2022, 2023, 2024, 2025]).preprocess_feats(
            force_recreate=model_args.force_recreate,
            force_recreate_preprocessing=model_args.force_recreate_preprocessing,
            clear_log=model_args.clear_log
        )
    
    model = XGBoostModel(model_args, logger, all_data)
    model.train_and_eval_model()
    test_pred = model.predict()
    

if __name__ == "__main__":
    main()
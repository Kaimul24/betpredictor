from pandas.core.api import DataFrame as DataFrame
import pandas as pd
import argparse, json
import xgboost as xgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict
from tqdm import tqdm

from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.models.two_stage import (
    MARKET_PROBABILITY_COL,
    add_pretrained_logit_feature,
)
from src.data.features.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT, XGBoostConfig
from src.utils import setup_logging, TupleAction

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

BASEBALL_ONLY_MODEL_NAME = "saved_xgboost_baseball_only.json"
FINAL_MODEL_NAME = "saved_xgboost.json"

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XGBoost Model Pipeline")
    parser.add_argument("--retune", action='store_true', help='Retune hyperparameters')
    parser.add_argument("--force-recreate", action="store_true", help="Recreate rolling features, even if cached file exists")
    parser.add_argument("--force-recreate-preprocessing", action="store_true", help="Recreate preprocessed datasets, even if cached file exists")
    parser.add_argument("--perspective-duplication", action="store_true", help="Duplicate training rows from each focal team's perspective")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    parser.add_argument("--batter-halflives", nargs='*', type=int, action=TupleAction, default=(4, 12), help="EWM halflives for batting stats")
    parser.add_argument("--starter-halflives", nargs='*', type=int, action=TupleAction, default=(3, 8), help="EWM halflives for starting pitching stats")
    parser.add_argument("--reliever-halflives", nargs='*', type=int, action=TupleAction, default=(3, 8), help="EWM halflives for relief pitching stats")
    parser.add_argument("--team-halflives", nargs='*', type=int, action=TupleAction, default=(3, 8, 20), help="EWM halflives for team metric stats")
    parser.add_argument(
        "--training-mode",
        choices=["stacked", "market_residual", "baseball_only"],
        default="stacked",
        help=(
            "stacked pretrains a baseball-only model on 2016-2019, then finetunes "
            "a 2021-2025 market-residual model. Use baseball_only or "
            "market_residual to run a single stage directly."
        )
        )
    return parser.parse_args()

class XGBoostModel:

    def __init__(self, config: XGBoostConfig, all_data: Dict, logger=None, mkt_only: bool = False):
        if not isinstance(config, XGBoostConfig):
            raise TypeError("XGBoostModel requires an XGBoostConfig")
        if config.training_mode == "stacked":
            raise ValueError("XGBoostModel requires a stage-specific config, not stacked")

        self.config = config
        self.model_args = config
        self.mkt_only = mkt_only
        if logger == None:
            self.logger = setup_logging("xgboost_model", LOG_FILE)
        else:
            self.logger = logger

        if all_data == {}:
            raise ValueError(f" No data was supplied.")

        self.training_mode = config.training_mode
        self.stage = config.stage
        self.uses_market_base_margin = self.training_mode == "market_residual" and not self.mkt_only
        self.hyperparam_path = HYPERPARAM_DIR / f"xgboost_hyperparams_{self.stage}_{self.training_mode}.json"
        self.model_path = self._model_path()
        self.calibrator_path = self._calibrator_path()

        X_train = all_data['X_train'].copy()
        y_train = all_data['y_train'].copy()
        X_val = all_data['X_val'].copy()
        y_val = all_data['y_val'].copy()
        X_test = all_data['X_test'].copy()
        y_test = all_data['y_test'].copy()

        n_val = len(y_val)
        cutoff = int(0.5 * n_val)

        self.val_cutoff = cutoff
        self.p_mkt_train = None
        self.p_mkt_val = None
        self.p_mkt_test = None

        if self.uses_market_base_margin:
            self._require_market_probability_column(X_train, X_val, X_test)
            p_mkt_train = X_train[MARKET_PROBABILITY_COL].to_numpy()
            p_mkt_test = X_test[MARKET_PROBABILITY_COL].to_numpy()
            p_mkt_val = X_val[MARKET_PROBABILITY_COL].to_numpy()

            self.p_mkt_train = p_mkt_train
            self.p_mkt_val = p_mkt_val
            self.p_mkt_test = p_mkt_test

            X_train = X_train.drop(columns=[MARKET_PROBABILITY_COL])
            X_val = X_val.drop(columns=[MARKET_PROBABILITY_COL])
            X_test = X_test.drop(columns=[MARKET_PROBABILITY_COL])

            dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=XGBoostModel.logit(p_mkt_train))
            dtest = xgb.DMatrix(X_test, label=y_test, base_margin=XGBoostModel.logit(p_mkt_test))

            dval_es = xgb.DMatrix(X_val[:cutoff], label=y_val[:cutoff], base_margin=XGBoostModel.logit(p_mkt_val[:cutoff]))
            dcal = xgb.DMatrix(X_val[cutoff:], label=y_val[cutoff:], base_margin=XGBoostModel.logit(p_mkt_val[cutoff:]))
        else:
            X_train = X_train.drop(columns=[MARKET_PROBABILITY_COL], errors="ignore")
            X_val = X_val.drop(columns=[MARKET_PROBABILITY_COL], errors="ignore")
            X_test = X_test.drop(columns=[MARKET_PROBABILITY_COL], errors="ignore")

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            dval_es = xgb.DMatrix(X_val[:cutoff], label=y_val[:cutoff])
            dcal = xgb.DMatrix(X_val[cutoff:], label=y_val[cutoff:])

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test
        
        self.y_cal   = self.y_val[cutoff:]
        self.y_val_es = self.y_val[:cutoff]

        self.dtrain = dtrain
        self.dtest = dtest

        self.dval_es = dval_es
        self.dcal = dcal

    def _model_path(self):
        if self.training_mode == "baseball_only":
            return SAVED_MODEL_DIR / BASEBALL_ONLY_MODEL_NAME
        return SAVED_MODEL_DIR / FINAL_MODEL_NAME

    def _calibrator_path(self):
        if self.training_mode == "baseball_only":
            return CAL_DIR / "xgb_calibrator_baseball_only.json"
        return CAL_DIR / "xgb_calibrator.json"

    def _require_market_probability_column(self, *frames: DataFrame) -> None:
        for frame in frames:
            if MARKET_PROBABILITY_COL not in frame.columns:
                raise ValueError(
                    f"{MARKET_PROBABILITY_COL} is required for market_residual XGBoost training. "
                    "Use baseball_only mode for pretraining data without odds."
                )

    @staticmethod
    def logit(p, eps=1e-6):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))

    def train_and_eval_model(self):
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
                    'objective': 'binary:logistic',
                    'device': "cpu",
                    'tree_method': 'hist',
                    'eval_metric': 'logloss',
                    "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                    'seed': 42,
                    'nthread': 6, 
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
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
            tscv = TimeSeriesSplit(n_splits=4)

            folds = list(tscv.split(np.arange(len(self.X_train))))
            
            scores = []

            for i, (tr_idx, val_idx) in enumerate(folds):
                dtr = dtrain.slice(tr_idx)
                dvl = dtrain.slice(val_idx)

                bst = xgb.train(
                    params,
                    dtr,
                    num_boost_round=4000,
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

        return study.best_params

    def analyze_feature_importance(self, bst, X_train):
        importance = bst.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': importance.keys(),
            'gain': importance.values()
        }).sort_values('gain', ascending=False)
        
        self.logger.info(f" Top 20 Features by Gain:\n{importance_df.head(20)}")
        
        total_gain = importance_df['gain'].sum()
        importance_df['gain_pct'] = (importance_df['gain'] / total_gain) * 100
        self.logger.info(f" \nTop feature contributes: {importance_df.iloc[0]['gain_pct']:.1f}%")
        self.logger.info(f" Top 5 features contribute: {importance_df.head(5)['gain_pct'].sum():.1f}%")
        
        self.logger.info(f" \nTotal features used: {len(importance)} / {X_train.shape[1]}")
        
        return importance_df
    
    def _percentile_summary(self, p_hat: np.ndarray, label: str):
        q = np.percentile(p_hat, [1,5,25,50,75,95,99])
        self.logger.info(
            f" [{label}] pred percentiles -> "
            f" p1={q[0]:.3f}, p5={q[1]:.3f}, p25={q[2]:.3f}, "
            f" p50={q[3]:.3f}, p75={q[4]:.3f}, p95={q[5]:.3f}, p99={q[6]:.3f}"
        )
        tails_low = np.mean(p_hat < 0.30) * 100
        tails_high = np.mean(p_hat > 0.70) * 100
        self.logger.info(f"[{label}] tail mass -> <0.30: {tails_low:.2f}% | >0.70: {tails_high:.2f}%")

    def _plot_logit_delta(self, p_hat: np.ndarray, p_mkt: np.ndarray, split: str):
        """
        Plot (1) histogram of delta_logit = logit(p_hat) - logit(p_mkt)
             (2) scatter of delta_logit vs logit(p_mkt).
        """
        z_hat = XGBoostModel.logit(p_hat)
        z_mkt = XGBoostModel.logit(p_mkt)
        delta = z_hat - z_mkt

        # 1) histogram
        plt.figure(figsize=(6,4))
        plt.hist(delta, bins=50, edgecolor="k", alpha=0.8)
        plt.xlabel("logit(pred) - logit(p_mkt)")
        plt.ylabel("Count")
        plt.title(f"Delta-Logit Histogram ({split})")
        out = PLOTS_DIR / f"delta_logit_hist_{split}.png"
        plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
        self.logger.info(f" Saved delta-logit histogram to {out}")

        # 2) scatter vs baseline logit
        plt.figure(figsize=(6,4))
        plt.scatter(z_mkt, delta, s=6, alpha=0.5)
        plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
        plt.xlabel("logit(p_mkt)")
        plt.ylabel("logit(pred) - logit(p_mkt)")
        plt.title(f"Delta-Logit vs Market ({split})")
        out = PLOTS_DIR / f"delta_logit_scatter_{split}.png"
        plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
        self.logger.info(f" Saved delta-logit scatter to {out}")

        # quick numeric summary
        self.logger.info(
            f" [{split}] delta logit mean={delta.mean():.4f}, std={delta.std():.4f}, "
            f" p5={np.percentile(delta,5):.4f}, p95={np.percentile(delta,95):.4f}"
        )


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
                    'objective': 'binary:logistic',
                    'device': "cpu",
                    'tree_method': 'hist',
                    'eval_metric': 'logloss',
                    'seed': 42,
                    'nthread': 12, 
            }
        
        all_params = {**BASE_PARAMS, **(hyperparams or {})}

        self.logger.info(" Training XGBoost model on CPU...")
        
        bst = xgb.train(
            all_params,
            self.dtrain,
            num_boost_round=4000,
            evals=[(self.dtrain, 'train'), (self.dval_es, 'eval')],
            verbose_eval=10,
            early_stopping_rounds=50
        )

        self.logger.info(f" Early stopping scores...")
        p_es = bst.predict(self.dval_es, iteration_range=(0, bst.best_iteration + 1), training=True)
        
        ll_es = log_loss(self.y_val_es, p_es)
        auc_es = roc_auc_score(self.y_val_es, p_es)
        brier_es = brier_score_loss(self.y_val_es, p_es)
        
        self.logger.info(f" Val(ES) Log Loss: {ll_es}")
        self.logger.info(f" Val(ES) ROC AUC: {auc_es}")
        self.logger.info(f" Val(ES) Brier:   {brier_es}")

        filepath = PLOTS_DIR / 'xgboost_calibration_es'
        out_path_curve = plot_calibration(self.y_val_es, p_es, split="val_es", filepath=filepath, n_bins=25)
        self.logger.info(f" Saved calibration curve (\"val_es\") to {out_path_curve}")

        self._percentile_summary(p_es, label="val_es (pre-cal)")
        if self.uses_market_base_margin:
            self._plot_logit_delta(p_es, self.p_mkt_val[:self.val_cutoff], split="val_es")

        self.logger.info(" Predictions on calibration set...")
        p_cal_raw = bst.predict(self.dcal, iteration_range=(0, bst.best_iteration + 1))

        ll_cal_pre    = log_loss(self.y_cal, p_cal_raw)
        auc_cal_pre   = roc_auc_score(self.y_cal, p_cal_raw)
        brier_cal_pre = brier_score_loss(self.y_cal, p_cal_raw)

        self.logger.info(f" Val(CAL) Pre-Calibration Log Loss: {ll_cal_pre}")
        self.logger.info(f" Val(CAL) Pre-Calibration ROC AUC: {auc_cal_pre}")
        self.logger.info(f" Val(CAL) Pre-Calibration Brier:   {brier_cal_pre}")

        self._percentile_summary(p_cal_raw, label="val_cal (pre-cal)")
        if self.uses_market_base_margin:
            self._plot_logit_delta(p_cal_raw, self.p_mkt_val[self.val_cutoff:], split="val_cal_pre")

        min_bin_size = 100
        filepath = PLOTS_DIR / 'xgboost_calibration'
        out_path_curve = plot_calibration(self.y_cal, p_cal_raw, split="pre_cal", filepath=filepath, n_bins=25, min_bin_size=min_bin_size)

        if self.uses_market_base_margin:
            calibrator, metrics = select_and_fit_calibrator(
                y_cal=self.y_cal,
                p_cal=p_cal_raw,
                p_mkt_cal=self.p_mkt_val[self.val_cutoff:],
                methods=("platt", "temperature"),
                selection_metric="logloss",
                min_bin_size=min_bin_size,
                n_bins=25,
                strategy="quantile",
            )
            p_cal_calibrated = apply_calibration(calibrator, p_cal_raw, self.p_mkt_val[self.val_cutoff:])
        else:
            calibrator, metrics = select_and_fit_calibrator(
                y_cal=self.y_cal,
                p_cal=p_cal_raw,
                methods=("isotonic",),
                selection_metric="logloss",
                min_bin_size=min_bin_size,
                n_bins=25,
                strategy="quantile",
            )
            p_cal_calibrated = apply_calibration(calibrator, p_cal_raw)
        
        self.logger.info(f" Calibration candidates: {metrics}")
        self.logger.info(f" Best calibrator: {calibrator}")
        save_calibrator(calibrator, str(self.calibrator_path))

        self.logger.info(" Evaluating calibrated predictions on calibration set...")
        
        ll_cal_after = log_loss(self.y_cal, p_cal_calibrated)
        auc_cal_after = roc_auc_score(self.y_cal, p_cal_calibrated)
        brier_cal_after = brier_score_loss(self.y_cal, p_cal_calibrated)
        
        self.logger.info(f" Val(CAL) Post-Calibration Log Loss: {ll_cal_after}")
        self.logger.info(f" Val(CAL) Post-Calibration ROC AUC: {auc_cal_after}")
        self.logger.info(f" Val(CAL) Post-Calibration Brier:   {brier_cal_after}")
        ll_improvement = ll_cal_pre - ll_cal_after
        brier_improvement = brier_cal_pre - brier_cal_after
        
        self.logger.info(f" Log Loss improvement: {ll_improvement:.4f}")
        self.logger.info(f" Brier Score improvement: {brier_improvement:.4f}")

        self._plot_roc(self.y_cal, p_cal_calibrated, split="cal")
        out_path_curve = plot_calibration(self.y_cal, p_cal_calibrated, split="cal_after", filepath=filepath, n_bins=25, min_bin_size=min_bin_size)

        self.logger.info(f" Saved calibration curve (\"cal_after\") to {out_path_curve}")

        self._percentile_summary(p_cal_calibrated, label="val_cal (post-cal)")
        if self.uses_market_base_margin:
            self._plot_logit_delta(p_cal_calibrated, self.p_mkt_val[self.val_cutoff:], split="val_cal_post")

        combined_X_train = pd.concat([self.X_train, self.X_val])
        combined_y_train = pd.concat([self.y_train, self.y_val])

        if self.uses_market_base_margin:
            combined_p_mkt = np.concatenate([self.p_mkt_train, self.p_mkt_val])
            combined_dtrain = xgb.DMatrix(data=combined_X_train, label=combined_y_train, base_margin=XGBoostModel.logit(combined_p_mkt))
        else:
            combined_dtrain = xgb.DMatrix(data=combined_X_train, label=combined_y_train)

        self.best_interation = bst.best_iteration
        num_boost_round = max(1, self.best_interation + 1)
        
        final_bst = xgb.train(
            all_params,
            combined_dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=10,
        )

        self.logger.info(" Retrained on train + val data.")
        _ = self.analyze_feature_importance(final_bst, self.X_train)
        self._save_model(final_bst)

        return final_bst
    

    def _save_hyperparams(self, params: Dict) -> None:
        self.hyperparam_path.write_text(json.dumps(params, indent=4))

    def _load_hyperparameters(self) -> Dict:
        if self.hyperparam_path.exists():
            return json.loads(self.hyperparam_path.read_text())
        
        self.logger.warning(f" No hyperparameters found at {self.hyperparam_path}")
        return {}

    def predict(self, model: xgb.Booster = None):
        if not model:
            model = self.load_model()

        if model is None:
            self.logger.error(f" No model has been trained!")
            return
        
        self.logger.info(f" Predicting on test set")
        p_test_raw = model.predict(self.dtest, training=False)
        self.logger.info(f" Raw predictions before calibration: Min: {p_test_raw.min()}, Max: {p_test_raw.max()}, Std: {p_test_raw.std()}")
        
        try:
            cal = load_calibrator(str(self.calibrator_path))
            if self.uses_market_base_margin:
                p_test = apply_calibration(cal, p_test_raw, self.p_mkt_test)
            else:
                p_test = apply_calibration(cal, p_test_raw)
            self.logger.info(" Applied calibrator to test predictions.")
            self.logger.info(f" Predictions after calibration: Min: {p_test.min()}, Max: {p_test.max()}, Std: {p_test.std()}")
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

        self._percentile_summary(p_test, label="test (post-cal)")
        if self.uses_market_base_margin:
            self._plot_logit_delta(p_test, self.p_mkt_test, split="test")

        return p_test
    
    def _save_model(self, model: xgb.Booster) -> None:
        xgboost_model_path = self.model_path
        self.xgboost_model_path = xgboost_model_path
        model.save_model(str(xgboost_model_path))

        self.logger.info(f" Succesfully saved XGBoost model to {xgboost_model_path}")

    def load_model(self) -> xgb.Booster:
        model_path = self.model_path
        if model_path.exists():
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            self.logger.info(f" Succesfully loaded XGBoost model")
            return bst
        
        self.logger.warning(f" No XGBoost model found")
        return None

def config_for_training_mode(config: XGBoostConfig, training_mode: str, stage: str) -> XGBoostConfig:
    return config.for_stage(training_mode=training_mode, stage=stage)

def preprocess_for_xgboost(config: XGBoostConfig, training_mode: str, stage: str):
    stage_config = config_for_training_mode(config, training_mode=training_mode, stage=stage)
    feature_config = stage_config.to_feature_config(stage=stage, training_mode=training_mode)
    seasons = (
        PreProcessing.PRETRAIN_YEARS
        if training_mode == "baseball_only"
        else PreProcessing.FINETUNE_YEARS
    )
    return PreProcessing(seasons, config=feature_config, mkt_only=False).preprocess_feats(
        force_recreate=stage_config.force_recreate,
        force_recreate_preprocessing=stage_config.force_recreate_preprocessing,
        clear_log=stage_config.clear_log,
    )

def train_baseball_only_xgboost(config: XGBoostConfig, logger):
    stage_config = config_for_training_mode(config, training_mode="baseball_only", stage="pretrain")
    model_data, _ = preprocess_for_xgboost(stage_config, training_mode="baseball_only", stage="pretrain")
    model = XGBoostModel(stage_config, model_data, logger, mkt_only=False)
    return model.train_and_eval_model()

def add_pretrained_xgboost_logits(model_data: Dict, pretrained_model: xgb.Booster) -> Dict:
    feature_names = pretrained_model.feature_names

    def predict_proba(X):
        return pretrained_model.predict(xgb.DMatrix(X))

    return add_pretrained_logit_feature(
        model_data,
        feature_names,
        predict_proba,
        XGBoostModel.logit,
    )

def train_market_residual_xgboost(config: XGBoostConfig, logger, pretrained_model: xgb.Booster | None = None):
    stage_config = config_for_training_mode(config, training_mode="market_residual", stage="finetune")
    model_data, odds_data = preprocess_for_xgboost(stage_config, training_mode="market_residual", stage="finetune")
    if pretrained_model is not None:
        model_data = add_pretrained_xgboost_logits(model_data, pretrained_model)

    model = XGBoostModel(stage_config, model_data, logger, mkt_only=False)
    trained_model = model.train_and_eval_model()
    test_pred = model.predict(trained_model)
    return trained_model, test_pred, odds_data

def main():
    args = create_args()
    config = XGBoostConfig.from_namespace(
        argparse.Namespace(
            **vars(args),
            stage="pretrain" if args.training_mode == "baseball_only" else "finetune",
        )
    )
    logger = setup_logging("xgboost_model", LOG_FILE, args=args)
    logger.info("="*75 + "XGBOOST MODEL" + "="*75)

    if config.training_mode == "stacked":
        pretrained_model = train_baseball_only_xgboost(config, logger)
        train_market_residual_xgboost(config, logger, pretrained_model=pretrained_model)
    elif config.training_mode == "baseball_only":
        stage_config = config_for_training_mode(config, training_mode="baseball_only", stage="pretrain")
        model_data, _ = preprocess_for_xgboost(stage_config, training_mode="baseball_only", stage="pretrain")
        model = XGBoostModel(stage_config, model_data, logger, mkt_only=False)
        trained_model = model.train_and_eval_model()
        model.predict(trained_model)
    else:
        train_market_residual_xgboost(config, logger)

if __name__ == "__main__":
    main()

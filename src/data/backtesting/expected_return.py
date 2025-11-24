from pandas.core.api import DataFrame as DataFrame
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import argparse
import numpy as np

from src.data.feature_preprocessing import PreProcessing
from src.data.models.xgboost_model import XGBoostModel
from src.config import PROJECT_ROOT
from src.utils import setup_logging

load_dotenv()

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "expected_return.log"

class ExpectedReturn:
    def __init__(self, model_data: DataFrame, odds_data: DataFrame, logger, mkt_only: bool = False) -> None:
        self.data = model_data
        self.odds_data = odds_data
        self.logger = logger
        self.mkt_only = mkt_only

    def roi_calculation(self):
        """
        Orchestrates the ROI on the test set for $1 bets on a given policy.
        """
        pred_df = self._setup_model()
        roi = self._evaluate_bets(pred_df)
        self.logger.info(f" ROI: {roi}")

    @staticmethod
    def calc_ev(pred_prob, mkt_prob):
        """
        Function is applied to a each row of the predictions and added as a new column
        New columns: ev_home, ev_away
        """
        return (pred_prob / mkt_prob) - 1

    def _setup_model(self) -> DataFrame:
        """
        Initializes, loads in, and predicts on the test set of an existing model  
        """
        model = XGBoostModel(model_args=None, all_data=self.data, mkt_only=self.mkt_only)
        pred = model.predict()

        df = DataFrame(self.data['y_test']).copy()
        df['p_home'] = pred
        df['p_away'] = 1 - df['p_home']
        return df

    
    def _simple_policy(self, max_ev_df: DataFrame):
        """
        This policy bets on the maximum non-zero EV for a game. If there are no non-zero EV, then no bet will occur.
        Flags each row with True or False in new to_bet column
        """
        max_ev_df['to_bet'] = np.where(max_ev_df['max_ev'] > 0, True, False)
        return max_ev_df

    def _evaluate_bets(self, pred_df: DataFrame):
        """
        Calcuates the total return on the test set for a given policy
        """
        odds_df = self.odds_data.copy()

        group_cols = ['game_date', 'dh', 'home_team', 'away_team', 'game_id']

        cutoff_idx = int(len(pred_df) / 2)
        cutoff_date = pred_df.index.get_level_values('game_date')[cutoff_idx]

        odds = odds_df[odds_df.index.get_level_values('game_date') >= cutoff_date]

        odds_with_pred = pred_df.merge(
            odds,
            on=group_cols,
            how='right',
            validate='1:m'
        )

        odds_with_pred['ev_home'] = ExpectedReturn.calc_ev(odds_with_pred['p_home'], odds_with_pred['home_opening_prob_raw'])
        odds_with_pred['ev_away'] = ExpectedReturn.calc_ev(odds_with_pred['p_away'], odds_with_pred['away_opening_prob_raw'])
        odds_with_pred['max_ev'] = odds_with_pred[['ev_home', 'ev_away']].max(axis=1)

        odds_with_pred['max_ev_side'] = np.where(odds_with_pred['max_ev'] == odds_with_pred['ev_home'], 'home', 'away')
        
        max_ev_rows = (
            odds_with_pred.sort_values('max_ev', ascending=False)
            .reset_index()
            .drop_duplicates(subset=group_cols, keep='first')
            .set_index(group_cols)
        )

        def plot_max_ev_distribution(max_ev_rows: DataFrame, bins: int = 30, show: bool = True):
            if max_ev_rows.empty:
                raise ValueError("max_ev_rows is empty; nothing to plot.")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(max_ev_rows['max_ev'], bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(max_ev_rows['max_ev'].mean(), color='red', linestyle='--', label='Mean')
            ax.set_xlabel('Max Expected Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Max EV per Game')
            ax.legend()
            fig.tight_layout()
            if show:
                plt.show()
            return ax
        
        max_ev_metrics = f"\
                      Mean: {max_ev_rows['max_ev'].mean()}\n\
                      Min: {max_ev_rows['max_ev'].min()}\n\
                      Max: {max_ev_rows['max_ev'].max()}\n\
                      Std: {max_ev_rows['max_ev'].std()}"
        
        self.logger.info(f" Max EV Metrics: \n{max_ev_metrics}")
   
        bet_decisions = self._simple_policy(max_ev_rows)
        bet_decisions = bet_decisions[bet_decisions['to_bet']]

        d_sel = np.where(bet_decisions['max_ev_side'] == 'home', bet_decisions['home_opening_prob_raw'], bet_decisions['away_opening_prob_raw'])
        y_sel = np.where(bet_decisions['max_ev_side'] == 'home', bet_decisions['is_winner_home'], 1 - bet_decisions['is_winner_home'])

        total_profit = (y_sel*(1.0/d_sel - 1.0) - (1 - y_sel)).sum()
        total_stake = float(len(bet_decisions))

        roi = total_profit / total_stake
        return roi

def create_args():
    parser = argparse.ArgumentParser(description="Expected return on bets runner")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    return parser.parse_args()

def main():
    args = create_args()
    logger = setup_logging('expected_return', log_file=LOG_FILE, args=args)

    mkt_only = False
    model_data, odds_data = PreProcessing([2021, 2022, 2023, 2024, 2025], model_type='xgboost', mkt_only=mkt_only).preprocess_feats()

    exp_ret = ExpectedReturn(model_data=model_data, odds_data=odds_data, logger=logger, mkt_only=mkt_only)

    exp_ret.roi_calculation()

if __name__ == "__main__":
    main()
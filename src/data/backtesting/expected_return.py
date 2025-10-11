from pandas.core.api import DataFrame as DataFrame
from dotenv import load_dotenv
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

"""
COMPUTE EXPECTED RETURN FOR BETTING $1 ON THE HIGHEST POSITIVE EXPECTED VALUE MONEYLINE ON 2025 SEASON

1. Call PreProcessing().preprocess_feats to get model data and odds data
    - model data will contain 2025 data
2. Initialize XGBoostModel with model_data
4. Call model.predict to predict on test set
    - output DataFrame/series? will be phome
5. Add paway = 1 - phome to each row of the predictions
6. For each sportsbook for each game, compute EV
7. Once all EV for each sportsbook has been calucated, pick highest EV
8. Once all picks are done, compare with what actually happened in the game to determine return over that period
    - If correct, add decimal odds d to total
    - If incorrect, subtract 1 from total

First implement a simple policy - Max non zero EV, then do more complex ones - https://chatgpt.com/g/g-p-6840b1d0b0f881918a6ece0e66a78da9-baseball-bet-predictor/c/68e98ee3-9ac4-8324-8803-095187f952a6?model=gpt-5-thinking

EV calculation: EV = (pred_prob / implied_prob) - 1

"""

class ExpectedReturn:
    def __init__(self, model_data: DataFrame, odds_data: DataFrame) -> None:
        self.data = model_data
        self.odds_data = odds_data

    def roi_calculation(self):
        """
        Orchestrates the ROI on the test set for $1 bets on a given policy.
        """
        pred = self._setup_model()

    @staticmethod
    def calc_ev(pred_prob):
        """
        Function is applied to a each row of the predictions and added as a new column
        New columns: ev_home, ev_away
        """
        pass

    def _setup_model(self) -> np.ndarray:
        """
        Initializes, loads in, and predicts on the test set of an existing model  
        """
        model = XGBoostModel(model_args=None, all_data=self.data)
        pred = model.predict()

        print(pred.max())
        print(self.data['y_test'])
    
    def _simple_policy():
        """
        This policy bets on the maximum non-zero EV for a game. If there are no non-zero EV, then no bet will occur.
        Flags each row with True or False in new to_bet column
        """
        pass

    def _evaluate_bets():
        """
        Calcuates the total return on the test set for a given policy
        """
        pass

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Expected return on bets runner")
    # parser.add_argument("--force-recreate", action="store_true", help="Recreate rolling features, even if cached file exists")
    # parser.add_argument("--force-recreate-preprocessing", action="store_true", help="Recreate preprocessed datasets, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    return parser.parse_args()

def main():
    args = create_args()
    logger = setup_logging('expected_return', log_file=LOG_FILE, args=args)

    model_data, odds_data = PreProcessing([2021, 2022, 2023, 2024, 2025]).preprocess_feats(
            is_xgboost=True
    )

    exp_ret = ExpectedReturn(model_data=model_data, odds_data=odds_data)
    exp_ret.logger = logger

    exp_ret.roi_calculation()

if __name__ == "__main__":
    main()
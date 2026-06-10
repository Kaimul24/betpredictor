import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb

from src.config import FeatureConfig, PROJECT_ROOT, TwoHeadNNConfig
from src.data.features.feature_preprocessing import PreProcessing
from src.data.models.calibration import apply_calibration, load_calibrator
from src.data.models.two_head_nn import DatasetFineTune, TwoHeadNN, get_device, logit
from src.data.models.two_stage import (
    MARKET_PROBABILITY_COL,
    PRETRAINED_BASEBALL_LOGIT_COL,
    evaluate_market_baseline,
    evaluate_probability_predictions,
    get_market_baseline_predictions,
)
from src.data.models.xgboost_model import XGBoostModel

SAVED_MODEL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_models"
CALIBRATOR_DIR = PROJECT_ROOT / "src" / "data" / "models" / "calibrators"

DEFAULT_MODELS = ("market_baseline", "xgboost", "two_head_nn")
DEFAULT_XGBOOST_PATH = SAVED_MODEL_DIR / "saved_xgboost.json"
DEFAULT_XGBOOST_BASEBALL_PATH = SAVED_MODEL_DIR / "saved_xgboost_baseball_only.json"
DEFAULT_TWO_HEAD_CHECKPOINT = SAVED_MODEL_DIR / "nn_finetune_ckpts" / "best_model.pt"
DEFAULT_XGBOOST_CALIBRATOR = CALIBRATOR_DIR / "xgb_calibrator.json"


@dataclass(frozen=True)
class PredictionResult:
    model: str
    split: str
    predictions: pd.Series
    artifact: str
    calibrated: bool


def create_args():
    parser = argparse.ArgumentParser(description="Compare saved model probabilities on finetune splits.")
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Models to compare.",
    )
    parser.add_argument("--xgboost-path", type=Path, default=DEFAULT_XGBOOST_PATH)
    parser.add_argument("--xgboost-baseball-path", type=Path, default=DEFAULT_XGBOOST_BASEBALL_PATH)
    parser.add_argument("--two-head-checkpoint", type=Path, default=DEFAULT_TWO_HEAD_CHECKPOINT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--force-recreate-preprocessing", action="store_true")
    parser.add_argument("--structured-player-features", action="store_true", help="Load structured NN player features for two-head NN predictions")
    return parser.parse_args()


def requested_splits(split: str) -> list[str]:
    return ["val", "test"] if split == "both" else [split]


def load_finetune_model_data(
    model_type: str,
    force_recreate: bool = False,
    force_recreate_preprocessing: bool = False,
    structured_player_features: bool = False,
):
    config = FeatureConfig(
        stage="finetune",
        training_mode="market_residual",
        model_type=model_type,
        structured_player_features=structured_player_features,
        perspective_duplication=False,
        force_recreate=force_recreate,
        force_recreate_preprocessing=force_recreate_preprocessing,
    )
    data, _ = PreProcessing(
        PreProcessing.FINETUNE_YEARS,
        config=config,
    ).preprocess_feats(
        force_recreate=force_recreate,
        force_recreate_preprocessing=force_recreate_preprocessing,
    )
    return data


def load_xgboost_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model artifact not found: {path}")

    booster = xgb.Booster()
    booster.load_model(str(path))
    if not booster.feature_names:
        raise ValueError(f"{path} does not contain feature names.")
    return booster


def prepare_xgboost_model_data_for_booster(
    model_data: dict,
    booster: xgb.Booster,
    baseball_booster: xgb.Booster | None = None,
) -> dict:
    feature_names = list(booster.feature_names or [])
    if PRETRAINED_BASEBALL_LOGIT_COL not in feature_names:
        return model_data

    if baseball_booster is None:
        raise ValueError(
            f"{PRETRAINED_BASEBALL_LOGIT_COL} is required by the final XGBoost model, "
            "but no baseball-only booster was supplied."
        )
    return add_pretrained_logits_for_saved_booster(model_data, baseball_booster)


def add_pretrained_logits_for_saved_booster(model_data: dict, baseball_booster: xgb.Booster) -> dict:
    feature_names = list(baseball_booster.feature_names or [])
    if not feature_names:
        raise ValueError("Baseball-only XGBoost artifact does not contain feature names.")

    updated = model_data.copy()
    for key in ["X_train", "X_val", "X_test"]:
        if key not in model_data:
            continue

        X = model_data[key].copy()
        missing = sorted(set(feature_names) - set(X.columns))
        if missing:
            raise ValueError(f"Baseball-only XGBoost {key} features are missing columns: {missing}")

        predictions = baseball_booster.predict(xgb.DMatrix(X.loc[:, feature_names]))
        X[PRETRAINED_BASEBALL_LOGIT_COL] = XGBoostModel.logit(predictions)
        updated[key] = X
    return updated


def xgboost_feature_frame(model_data: dict, booster: xgb.Booster, split: str) -> pd.DataFrame:
    key = f"X_{split}"
    if key not in model_data:
        raise KeyError(f"model_data is missing {key}.")

    feature_names = list(booster.feature_names or [])
    X = model_data[key]
    missing = sorted(set(feature_names) - set(X.columns))
    if missing:
        raise ValueError(
            f"Saved XGBoost artifact is incompatible with current {split} preprocessing. "
            f"Missing columns: {missing}. Retrain the artifact with the current feature contract."
        )
    return X.loc[:, feature_names]


def predict_xgboost_split(
    model_data: dict,
    split: str,
    xgboost_path: Path,
    baseball_path: Path,
    calibrator_path: Path = DEFAULT_XGBOOST_CALIBRATOR,
) -> PredictionResult:
    booster = load_xgboost_booster(xgboost_path)
    baseball_booster = None
    if PRETRAINED_BASEBALL_LOGIT_COL in list(booster.feature_names or []):
        baseball_booster = load_xgboost_booster(baseball_path)

    prepared_data = prepare_xgboost_model_data_for_booster(
        model_data,
        booster,
        baseball_booster=baseball_booster,
    )
    p_market = get_market_baseline_predictions(prepared_data, split=split)
    X = xgboost_feature_frame(prepared_data, booster, split)
    dmatrix = xgb.DMatrix(X, base_margin=XGBoostModel.logit(p_market))
    predictions = booster.predict(dmatrix, training=False)
    calibrated = False

    if calibrator_path.exists():
        calibrator = load_calibrator(str(calibrator_path))
        predictions = apply_calibration(calibrator, predictions, p_market)
        calibrated = True

    return PredictionResult(
        model="xgboost",
        split=split,
        predictions=pd.Series(predictions, index=prepared_data[f"X_{split}"].index),
        artifact=str(xgboost_path),
        calibrated=calibrated,
    )


def load_two_head_model_from_checkpoint(
    checkpoint_path: Path,
    in_dim: int,
    device: torch.device,
) -> tuple[TwoHeadNN, TwoHeadNNConfig]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Two-head NN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise RuntimeError(f"{checkpoint_path} is missing checkpoint config metadata.")
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"{checkpoint_path} is missing a model state_dict.")

    config = TwoHeadNNConfig(**checkpoint_config)
    model = TwoHeadNN(
        in_dim=in_dim,
        base_hidden_size=config.base_hidden_size,
        p_drop=config.p_drop,
        max_residual=config.max_residual,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def predict_two_head_split(
    model_data: dict,
    split: str,
    checkpoint_path: Path,
    device_name: str = "auto",
) -> PredictionResult:
    X = model_data[f"X_{split}"]
    y = model_data[f"y_{split}"]
    dataset = DatasetFineTune(
        X,
        y,
        structured_player_features=model_data.get(f"X_{split}_structured_flat"),
    )
    device = get_device(device_name)
    model, config = load_two_head_model_from_checkpoint(
        checkpoint_path,
        in_dim=dataset.X.shape[1],
        device=device,
    )

    loader = DataLoader(dataset, batch_size=config.val_batch, shuffle=False, num_workers=0)
    predictions = []
    for features, _, p_market in loader:
        features = features.to(device)
        p_market = p_market.to(device)
        logits = logit(p_market.view(-1, 1)) + model.forward_finetune(features, alpha=config.alpha)
        predictions.extend(torch.sigmoid(logits).view(-1).cpu().numpy())

    return PredictionResult(
        model="two_head_nn",
        split=split,
        predictions=pd.Series(predictions, index=X.index),
        artifact=str(checkpoint_path),
        calibrated=False,
    )


def market_baseline_result(model_data: dict, split: str) -> PredictionResult:
    return PredictionResult(
        model="market_baseline",
        split=split,
        predictions=pd.Series(
            get_market_baseline_predictions(model_data, split=split),
            index=model_data[f"X_{split}"].index,
        ),
        artifact=MARKET_PROBABILITY_COL,
        calibrated=False,
    )


def assert_prediction_index(result: PredictionResult, expected_index: pd.Index) -> None:
    if not result.predictions.index.equals(expected_index):
        raise ValueError(
            f"{result.model} predictions for {result.split} do not match the canonical split index."
        )


def build_comparison_row(
    result: PredictionResult,
    y_true,
    market_metrics: dict[str, float],
) -> dict[str, float | int | str | bool]:
    metrics = evaluate_probability_predictions(y_true, result.predictions.to_numpy())
    predictions = result.predictions.to_numpy(dtype=float)
    return {
        "model": result.model,
        "split": result.split,
        "n": len(predictions),
        "log_loss": metrics["log_loss"],
        "delta_log_loss_vs_market": metrics["log_loss"] - market_metrics["log_loss"],
        "brier": metrics["brier"],
        "delta_brier_vs_market": metrics["brier"] - market_metrics["brier"],
        "roc_auc": metrics["roc_auc"],
        "mean_pred": float(np.mean(predictions)),
        "std_pred": float(np.std(predictions)),
        "calibrated": result.calibrated,
        "artifact": str("/".join(Path(result.artifact).parts[-3:]))
    }


def compare_predictions(
    canonical_data: dict,
    split: str,
    results: list[PredictionResult],
) -> pd.DataFrame:
    expected_index = canonical_data[f"X_{split}"].index
    market_metrics = evaluate_market_baseline(canonical_data, split=split)
    rows = []
    for result in results:
        assert_prediction_index(result, expected_index)
        rows.append(build_comparison_row(result, canonical_data[f"y_{split}"], market_metrics))

    return pd.DataFrame(rows).sort_values(["split", "log_loss", "model"]).reset_index(drop=True)


def main():
    args = create_args()
    splits = requested_splits(args.split)
    xgboost_data = load_finetune_model_data(
        model_type="xgboost",
        force_recreate=args.force_recreate,
        force_recreate_preprocessing=args.force_recreate_preprocessing,
        structured_player_features=False,
    )
    nn_data = None
    if "two_head_nn" in args.models:
        nn_data = load_finetune_model_data(
            model_type="mlp",
            force_recreate=args.force_recreate,
            force_recreate_preprocessing=args.force_recreate_preprocessing,
            structured_player_features=args.structured_player_features,
        )

    frames = []
    for split in splits:
        results = []
        if "market_baseline" in args.models:
            results.append(market_baseline_result(xgboost_data, split))
        if "xgboost" in args.models:
            results.append(
                predict_xgboost_split(
                    xgboost_data,
                    split=split,
                    xgboost_path=args.xgboost_path,
                    baseball_path=args.xgboost_baseball_path,
                )
            )
        if "two_head_nn" in args.models:
            results.append(
                predict_two_head_split(
                    nn_data,
                    split=split,
                    checkpoint_path=args.two_head_checkpoint,
                    device_name=args.device,
                )
            )
        frames.append(compare_predictions(xgboost_data, split, results))

    comparison = pd.concat(frames, ignore_index=True)
    print(comparison.to_string(index=False))
    if args.output_csv is not None:
        comparison.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

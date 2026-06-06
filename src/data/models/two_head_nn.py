from pandas.core.api import DataFrame as DataFrame
import pandas as pd
from typing import List
import pickle
import optuna, json, argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Callable
from dataclasses import replace

from src.data.models.two_stage import MARKET_PROBABILITY_COL, baseball_feature_columns
from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.features.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT, TwoHeadNNConfig
from src.utils import setup_logging, TupleAction

SAVED_MODEL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_models"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

HYPERPARAM_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_hyperparameters"
HYPERPARAM_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_HYPERPARAM_PATH = HYPERPARAM_DIR / "two_head_pretrain_hyperparams.json"
PRETRAIN_HYPERPARAM_KEYS = ("base_hidden_size", "p_drop", "lr", "weight_decay")

PRETRAIN_CHECKPOINTS = SAVED_MODEL_DIR / "nn_pretrain_ckpts"
PRETRAIN_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

FINETUNE_CHECKPOINTS = SAVED_MODEL_DIR / "nn_finetune_ckpts"
FINETUNE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "two_head_nn.log"

PLOTS_DIR = PROJECT_ROOT / "src" / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Network Model Pipeline")
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
    parser.add_argument(
        "--pretrain-dir",
        type=str,
        default=PRETRAIN_CHECKPOINTS,
        help="Path to pretrained model directory (baseball only)."
    )
    parser.add_argument(
        "--finetune-dir",
        type=str,
        default=FINETUNE_CHECKPOINTS,
        help="Path to market residual model directory (pretrain + finetune)."
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=PRETRAIN_CHECKPOINTS / "best_model.pt",
        help="Optional pretrained checkpoint file. Defaults to --pretrain-dir/best_model.pt."
    )
    parser.add_argument("--device", type=str, default="auto", help="Training device.")
    parser.add_argument("--base-hidden-size", type=int, default=32, help="Base hidden units per layer")
    parser.add_argument("--max-residual", type=float, default=0.5, help="Max market adjustment")
    parser.add_argument("--alpha", type=float, default=0.7, help="Adjustment scaling factor")
    parser.add_argument("--p-drop", type=float, default=0.2, help="Dropout Probability")
    parser.add_argument("--train-batch", type=int, default=256, help="Training Batch Size")
    parser.add_argument("--val-batch", type=int, default=512, help="Val Batch Size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight-decay", type=float, default=0.03, help="AdamW weight decay")
    parser.add_argument("--retune", action="store_true", help="Tune pretrain hyperparameters with Optuna")
    parser.add_argument("--use-hyperparams", action="store_true", help="Use saved pretrain hyperparameters")
    parser.add_argument("--pretrain-trials", type=int, default=50, help="Number of Optuna pretrain tuning trials")
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
    return parser.parse_args()

class TwoHeadNN(nn.Module):
    def __init__(self, in_dim: int, base_hidden_size: int = 256, p_drop: float = 0.2, max_residual: float = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.base_hidden_size = base_hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, base_hidden_size),
            nn.ReLU(),
            
            nn.Linear(base_hidden_size, base_hidden_size ),
            nn.Dropout(p_drop),
            nn.ReLU(),

            nn.Linear(base_hidden_size, base_hidden_size),
            nn.Dropout(p_drop),
            nn.ReLU(),

            nn.Linear(base_hidden_size, base_hidden_size),
            nn.Dropout(p_drop),
            nn.ReLU(),

            nn.Linear(base_hidden_size, base_hidden_size),
            nn.Dropout(p_drop),
            nn.ReLU()
        )

        self.baseball_head = nn.Linear(base_hidden_size, 1)
        self.market_residual = nn.Linear(base_hidden_size, 1)
        self.max_residual = max_residual

    def baseball_logit(self, x):
        z = self.encoder(x)
        return self.baseball_head(z)

    def residual_logit(self, x):
        z = self.encoder(x)
        return self.max_residual * torch.tanh(self.market_residual(z))

    def forward_pretrain(self, x):
        return self.baseball_logit(x)
    
    def forward_finetune(self, x, alpha: float = 1.0):
        return alpha * self.residual_logit(x)
    
class DatasetPretrain(Dataset):
    def __init__(self, X: DataFrame, y: pd.Series, mkt_col: str = MARKET_PROBABILITY_COL):
        assert mkt_col not in X.columns
        assert len(X) == len(y)
        self.X = X.to_numpy()
        self.y = y.to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return feature, label

class DatasetFineTune(Dataset):
    def __init__(self, X: DataFrame, y: pd.Series, mkt_col: str = MARKET_PROBABILITY_COL):
        assert mkt_col in X.columns
        assert len(X) == len(y)
        self.feature_columns = baseball_feature_columns(X.columns)
        self.X = X.loc[:, self.feature_columns].to_numpy()
        self.y = y.to_numpy()
        self.p_mkt = X[mkt_col].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        p_mkt = torch.tensor(self.p_mkt[idx], dtype=torch.float32)
        return feature, label, p_mkt

def get_device(dev: str) -> torch.device:
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev)

def _set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(in_dim: int, config: TwoHeadNNConfig, device: torch.device) -> TwoHeadNN:
    return TwoHeadNN(
        in_dim=in_dim,
        base_hidden_size=config.base_hidden_size,
        p_drop=config.p_drop,
        max_residual=config.max_residual,
    ).to(device)

def build_optimizer(model: TwoHeadNN, config: TwoHeadNNConfig) -> torch.optim.Optimizer:
    return optim.AdamW(model.parameters(), weight_decay=config.weight_decay, lr=config.lr)

def build_scheduler(optimizer: torch.optim.Optimizer, config: TwoHeadNNConfig):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr,
    )

def _coerce_pretrain_hyperparameters(params: dict) -> dict:
    missing = [key for key in PRETRAIN_HYPERPARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"Missing pretrain hyperparameters: {missing}")
    return {
        "base_hidden_size": int(params["base_hidden_size"]),
        "p_drop": float(params["p_drop"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
    }

def apply_pretrain_hyperparameters(config: TwoHeadNNConfig, params: dict) -> TwoHeadNNConfig:
    return replace(config, **_coerce_pretrain_hyperparameters(params))

def save_pretrain_hyperparameters(
    params: dict,
    best_value: float,
    n_trials: int,
    path: Path = PRETRAIN_HYPERPARAM_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": _coerce_pretrain_hyperparameters(params),
        "best_value": float(best_value),
        "n_trials": int(n_trials),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_pretrain_hyperparameters(path: Path = PRETRAIN_HYPERPARAM_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Saved pretrain hyperparameters not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("best_params", payload)
    return _coerce_pretrain_hyperparameters(params)

def pretrain_stability_objective(val_losses: list[float], window: int = 5) -> float:
    if not val_losses:
        raise ValueError("val_losses must contain at least one epoch")
    if window < 1:
        raise ValueError("window must be at least 1")

    window = min(window, len(val_losses))
    best_epoch, best_val = min(enumerate(val_losses), key=lambda item: item[1])
    smoothed_vals = [
        sum(val_losses[i:i + window]) / window
        for i in range(len(val_losses) - window + 1)
    ]
    best_smoothed_val = min(smoothed_vals)
    tail_val = sum(val_losses[-window:]) / window
    overfit_penalty = max(0.0, tail_val - best_val)
    min_good_epoch = int(0.15 * len(val_losses))
    early_peak_penalty = max(0.0, (min_good_epoch - best_epoch) / len(val_losses))

    return (
        best_smoothed_val
        + 0.25 * overfit_penalty
        + 0.02 * early_peak_penalty
    )

def tune_pretrain_hyperparameters(
    config: TwoHeadNNConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_dim: int,
    device: torch.device,
    logger,
    n_trials: int | None = None,
    hyperparam_path: Path = PRETRAIN_HYPERPARAM_PATH,
) -> dict:
    trial_count = n_trials if n_trials is not None else config.pretrain_trials
    criterion = nn.BCEWithLogitsLoss().to(device)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "base_hidden_size": trial.suggest_categorical("base_hidden_size", [16, 32, 64]),
            "p_drop": trial.suggest_float("p_drop", 0.1, 0.6),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 2e-1, log=True),
        }
        trial_config = apply_pretrain_hyperparameters(config, params)
        _set_torch_seed(42)
        model = build_model(in_dim, trial_config, device)
        optimizer = build_optimizer(model, trial_config)
        scheduler = build_scheduler(optimizer, trial_config)
        train_losses, val_losses = [], []

        for epoch in range(trial_config.epochs):
            train_loss = train_epoch_pretrain(model, train_loader, optimizer, criterion, device)
            val_loss = val_epoch_pretrain(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return pretrain_stability_objective(val_losses)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    logger.info(f" Tuning pretrain hyperparameters for {trial_count} trials...")

    with tqdm(total=trial_count, desc="Optuna pretrain tuning") as progress:
        def _tick(study, trial):
            progress.update(1)

        study.optimize(objective, n_trials=trial_count, callbacks=[_tick])

    best_params = _coerce_pretrain_hyperparameters(study.best_params)
    save_pretrain_hyperparameters(
        params=best_params,
        best_value=study.best_value,
        n_trials=trial_count,
        path=hyperparam_path,
    )
    logger.info(f" Best pretrain stability objective: {study.best_value:.4f}")
    logger.info(f" Saved pretrain hyperparameters to {hyperparam_path}")
    return best_params

def prepare_data(
        stage: str,
        config: TwoHeadNNConfig,
        train_batch_size: int = 256,
        val_test_batch_size: int = 512
    ) -> tuple[DataLoader, DataLoader, DataLoader, int]:

    if not isinstance(config, TwoHeadNNConfig):
        raise TypeError("prepare_data requires a TwoHeadNNConfig")
    
    if stage == "pretrain":
        seasons = PreProcessing.PRETRAIN_YEARS
        dataset = DatasetPretrain
        training_mode = "baseball_only"
    elif stage == "finetune":
        seasons = PreProcessing.FINETUNE_YEARS
        dataset = DatasetFineTune
        training_mode = "market_residual"
    else:
        raise ValueError("Invalid stage. Must be 'pretrain' or 'finetune'")

    feature_config = config.to_feature_config(
        stage=stage,
        training_mode=training_mode,
        model_type="mlp",
    )
    preprocessing = PreProcessing(
        seasons=seasons, 
        config=feature_config,
    )
    data, _ = preprocessing.preprocess_feats(
            config.force_recreate,
            config.force_recreate_preprocessing,
            config.clear_log
    )
    train_dataset = dataset(
        data["X_train"],
        data["y_train"]
    )
    val_dataset = dataset(
        data["X_val"],
        data["y_val"]
    )
    test_dataset = dataset(
        data["X_test"],
        data["y_test"]
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=val_test_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=val_test_batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_loader.dataset.X.shape[1]

def save_checkpoint(
        model: TwoHeadNN, 
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        path: str | Path, 
        config: TwoHeadNNConfig
    ) -> None:

    torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.to_dict(),
        }, path
    )

def _load_checkpoint_file(model_path: str | Path, device: torch.device | str | None = None) -> dict:
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"{model_path} appears to be an old checkpoint that cannot be loaded safely. "
            "Delete it or retrain it so a dataclass-config checkpoint can be written."
        ) from exc

def load_checkpoint(
        model: TwoHeadNN, 
        model_path: str | Path,
        config: TwoHeadNNConfig,
        optimizer: torch.optim.Optimizer | None = None, 
        device: torch.device | str | None = None,
    ) -> tuple[TwoHeadNN, torch.optim.Optimizer | None]:
    """Resume an exact same-stage checkpoint, including config and optimizer."""
    ckpt = _load_checkpoint_file(model_path, device=device)

    checkpoint_config = ckpt.get("config")
    if checkpoint_config is None:
        raise RuntimeError(
            f"{model_path} is missing checkpoint config metadata. "
            "Delete it or retrain it with the dataclass config format."
        )
    if checkpoint_config != config.to_dict():
        raise RuntimeError(
            f"{model_path} was saved with a different config. "
            "Use load_pretrained_checkpoint for transfer learning, or resume with the original config."
        )
    
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    return model, optimizer

def load_pretrained_checkpoint(
        model: TwoHeadNN,
        model_path: str | Path,
        device: torch.device | str | None = None,
    ) -> TwoHeadNN:
    """Transfer-load pretrained weights from disk without optimizer/config coupling."""
    ckpt = _load_checkpoint_file(model_path, device=device)
    if "state_dict" not in ckpt:
        raise RuntimeError(f"{model_path} is missing a model state_dict.")

    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load pretrained checkpoint {model_path} into a model with "
            f"in_dim={model.in_dim}. Pretrain and finetune must use the same baseball "
            "feature columns; market columns should be kept only as the finetune baseline."
        ) from exc

    return model

def logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def train_epoch_pretrain(
        model: TwoHeadNN,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        device,
    ) -> float:

    model.train()
    total_loss = 0.0
    n_batches = 0

    for (X, y) in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_pretrain(X)
        loss = criterion(pred, y.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)

def train_epoch_finetune(
        model: TwoHeadNN,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        device,
        alpha: float = 1.0
    ) -> float:

    model.train()
    total_loss = 0.0
    n_batches = 0

    for (X, y, p_mkt) in loader:
        X, y, p_mkt = X.to(device), y.to(device), p_mkt.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_raw = model.forward_finetune(X, alpha=alpha)
        p_res = logit(p_mkt.reshape(-1,1)) + pred_raw
        loss = criterion(p_res, y.view(-1, 1))
        # res = pred_raw - p_mkt
        # loss += 0.05 * res.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)

@torch.no_grad()
def val_epoch_pretrain(
        model: TwoHeadNN,
        loader: DataLoader,
        criterion,
        device,
    ) -> float:

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (X, y) in loader:
        X, y = X.to(device), y.to(device)
        pred = model.forward_pretrain(X)
        loss = criterion(pred, y.view(-1, 1))
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)

@torch.no_grad()
def val_epoch_finetune(
        model: TwoHeadNN,
        loader: DataLoader,
        criterion,
        device,
        alpha: float = 1.0
    ) -> float:

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for (X, y, p_mkt) in loader:
        X, y, p_mkt = X.to(device), y.to(device), p_mkt.to(device)
        pred_raw = model.forward_finetune(X, alpha=alpha)
        pred_res = logit(p_mkt.view(-1,1)) + pred_raw
        loss = criterion(pred_res, y.view(-1, 1))
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)

def train(
    model: TwoHeadNN, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer, 
    criterion, 
    scheduler,
    device,
    train_epoch_fn: Callable,
    val_epoch_fn: Callable,
    checkpoint_dir: Path,
    epochs: int,
    logger,
    config: TwoHeadNNConfig
) -> tuple[list[float], list[float], float]:
    best_val = float("inf")
    train_losses, val_losses = [], []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(0, epochs), desc="Training Progress", leave=False):
        train_loss = train_epoch_fn(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = val_epoch_fn(
            model, val_loader, criterion, device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f" Epoch {epoch+1:3d}/{epochs}  "
            f" train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            logger.info(f"  New best model: {best_val:.4f}, saving...")
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, best_path, config=config
            )
        elif (epoch + 1) % 10 == 0:
            path = checkpoint_dir / f"epoch_{epoch+1:04d}.pt"
            logger.info(f" Saving checkpoint to {path}. Epoch: {epoch+1}")
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, path, config=config
            )

    return train_losses, val_losses, best_val

def pretrain(device, logger, config: TwoHeadNNConfig) -> TwoHeadNN:
    pt_config = config.for_stage(training_mode="baseball_only", stage="pretrain", perspective_duplication=True)
    pt_train_ldr, pt_val_ldr, pt_test_ldr, pt_in_dim = prepare_data(
        stage="pretrain",
        config=pt_config,
        train_batch_size=pt_config.train_batch,
        val_test_batch_size=pt_config.val_batch
    )

    if pt_config.retune:
        best_params = tune_pretrain_hyperparameters(
            config=pt_config,
            train_loader=pt_train_ldr,
            val_loader=pt_val_ldr,
            in_dim=pt_in_dim,
            device=device,
            logger=logger,
        )
        pt_config = apply_pretrain_hyperparameters(pt_config, best_params)
    elif pt_config.use_hyperparams:
        best_params = load_pretrain_hyperparameters()
        pt_config = apply_pretrain_hyperparameters(pt_config, best_params)
        logger.info(f" Loaded pretrain hyperparameters from {PRETRAIN_HYPERPARAM_PATH}")
            
    pretrained_model = build_model(pt_in_dim, pt_config, device)

    optimizer = build_optimizer(pretrained_model, pt_config)
    scheduler = build_scheduler(optimizer, pt_config)
    criterion = nn.BCEWithLogitsLoss().to(device)
    logger.info(
        " Pretraining for "
        f"{pt_config.epochs} epochs with hidden={pt_config.base_hidden_size}, "
        f"p_drop={pt_config.p_drop:.3f}, lr={pt_config.lr:.2e}, "
        f"weight_decay={pt_config.weight_decay:.2e}..."
    )

    train_losses_pt, val_losses_pt, best_val_pt = train(
        model=pretrained_model,
        train_loader=pt_train_ldr,
        val_loader=pt_val_ldr,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_epoch_fn=train_epoch_pretrain,
        val_epoch_fn=val_epoch_pretrain,
        checkpoint_dir=pt_config.pretrain_dir,
        epochs=pt_config.epochs,
        logger=logger,
        config=pt_config
    )

    logger.info(f" Done pretraining. Best val loss: {best_val_pt}")
    _plot_loss(train_losses_pt, val_losses_pt, "pretrain_plot.png", logger)

    best_model = build_model(pt_in_dim, pt_config, device)
    return load_pretrained_checkpoint(best_model, pt_config.pretrain_dir / "best_model.pt", device=device)
    

def finetune(
    device,
    logger,
    config: TwoHeadNNConfig,
    pretrained_model: TwoHeadNN | None = None,
) -> None:
    ft_config = config.for_stage(training_mode="market_residual", stage="finetune", perspective_duplication=False)
    ft_train_ldr, ft_val_ldr, ft_test_ldr, ft_in_dim = prepare_data(
        stage="finetune",
        config=ft_config,
        train_batch_size=ft_config.train_batch,
        val_test_batch_size=ft_config.val_batch
    )

    if pretrained_model is None:
        ft_model = build_model(ft_in_dim, ft_config, device)
        checkpoint_path = config.pretrained_checkpoint or (config.pretrain_dir / "best_model.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {checkpoint_path}. "
                "Run --training-mode baseball_only first or pass --pretrained-checkpoint."
            )
        logger.info(f" Loading pretrained checkpoint from {checkpoint_path}")
        load_pretrained_checkpoint(ft_model, checkpoint_path, device=device)
    else:
        if pretrained_model.in_dim != ft_in_dim:
            raise RuntimeError(
                f"Pretrained model input dimension {pretrained_model.in_dim} does not match "
                f"finetune input dimension {ft_in_dim}. Rebuild preprocessing caches so both "
                "stages use the same baseball feature columns."
            )
        ft_config = replace(ft_config, base_hidden_size=pretrained_model.base_hidden_size)
        ft_model = build_model(ft_in_dim, ft_config, device)
        ft_model.load_state_dict(pretrained_model.state_dict())

    optimizer = build_optimizer(ft_model, ft_config)
    scheduler = build_scheduler(optimizer, ft_config)
    criterion = nn.BCEWithLogitsLoss().to(device)
    logger.info(f" Finetuning for {ft_config.epochs} epochs...")

    train_losses_ft, val_losses_ft, best_val_ft = train(
        model=ft_model,
        train_loader=ft_train_ldr,
        val_loader=ft_val_ldr,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_epoch_fn=train_epoch_finetune,
        val_epoch_fn=val_epoch_finetune,
        checkpoint_dir=ft_config.finetune_dir,
        epochs=ft_config.epochs,
        logger=logger,
        config=ft_config
    )

    logger.info(f" Done finetuning. Best val loss: {best_val_ft}")
    _plot_loss(train_losses_ft, val_losses_ft, "finetune_plot.png", logger)

def _plot_loss(train_losses: List[float], val_losses: List[float], plot_name: str, logger) -> None:
        """Plot training and validation loss curves across epochs.
        
        Args:
            train_losses: List of average training losses per epoch
            val_losses: List of validation losses per epoch
        """
        if len(train_losses) != len(val_losses):
            raise ValueError("train_losses and val_losses must have the same length")
        
        if not train_losses:
            return
        epochs = range(1, len(train_losses) + 1)

        plt.figure("Training Loss Curves", figsize=(10, 6))
        plt.clf()
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = PLOTS_DIR / f"{plot_name}"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f" Loss curves saved to {plot_path}")

def validate_tuning_mode(config: TwoHeadNNConfig) -> None:
    if config.training_mode == "market_residual" and config.retune:
        raise ValueError(
            "--retune tunes only the baseball-only pretrain stage. "
            "Use --training-mode stacked or --training-mode baseball_only."
        )

def main():
    args = create_args()
    config = TwoHeadNNConfig.from_namespace(
        argparse.Namespace(
            **vars(args),
            stage="pretrain" if args.training_mode == "baseball_only" else "finetune",
        )
    )
    validate_tuning_mode(config)

    logger = setup_logging("two_head_nn", LOG_FILE, args=args)
    device = get_device(config.device)

    if config.training_mode == "stacked":
        logger.info(f" Stacked Training Mode")
        pretrained_model = pretrain(device, logger, config)
        finetune(device, logger, config, pretrained_model=pretrained_model)
    elif config.training_mode == "baseball_only":
        logger.info(f" Pretrain Training Mode (Baseball Only)")
        pretrain(device, logger, config)
    elif config.training_mode == "market_residual":
        logger.info(f" Finetune Training Mode (Market Residual)")
        finetune(device, logger, config)


if __name__ == "__main__":
    main()

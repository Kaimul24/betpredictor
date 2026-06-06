"""
DEPRECATED, use two_head_nn.py
"""
from pandas.core.api import DataFrame as DataFrame
import pandas as pd
from typing import Dict, List
import optuna, json, argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.features.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT, NeuralNetworkConfig
from src.utils import setup_logging, TupleAction

HYPERPARAM_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_hyperparameters"
HYPERPARAM_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_models"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "neural_network_log.log"

PLOTS_DIR = PROJECT_ROOT / "src" / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Network Model Pipeline")
    parser.add_argument(
        "--training-mode",
        choices=["market_residual"],
        default="market_residual",
        help="Train the legacy MLP as a market-residual model."
        )
    parser.add_argument("--base-hidden-size", type=int, default=256, help="Base hidden units per layer")
    parser.add_argument("--max-residual", type=float, default=0.5, help="Max market adjustment")
    parser.add_argument("--alpha", type=float, default=0.7, help="Adjustment scaling factor")
    parser.add_argument("--p-drop", type=float, default=0.2, help="Dropout Probability")
    parser.add_argument("--train-batch", type=int, default=1024, help="Training Batch Size")
    parser.add_argument("--val-batch", type=int, default=8192, help="Val Batch Size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--retune", action="store_true", help="Retune hyperparameters")
    parser.add_argument("--use-hyperparams", action="store_true", help="Use saved hyperparameters")
    parser.add_argument(
        "--cosine-scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anneal learning rate with CosineAnnealingLR over training epochs",
    )
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

def prepare_mlp_data(
        config: NeuralNetworkConfig,
        model_data: Dict[str, DataFrame] | None = None, 
    ) -> Dict[str, int | DataLoader]:
    if not isinstance(config, NeuralNetworkConfig):
        raise TypeError("prepare_mlp_data requires a NeuralNetworkConfig")

    def to_tensor(x: DataFrame) -> torch.Tensor:
        return torch.tensor(x.values, dtype=torch.float32)
    
    def logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)
    
    if not model_data:
        feature_config = config.to_feature_config(stage="finetune", training_mode="market_residual")
        model_data, _ = PreProcessing(
            PreProcessing.FINETUNE_YEARS,
            config=feature_config,
            mkt_only=False,
        ).preprocess_feats(
            force_recreate=config.force_recreate,
            force_recreate_preprocessing=config.force_recreate_preprocessing,
            clear_log=config.clear_log,
        )

    p_mkt_train = model_data["X_train"]["p_open_home_median_nv"].to_numpy()
    p_mkt_test = model_data["X_test"]["p_open_home_median_nv"].to_numpy()
    p_mkt_val = model_data["X_val"]["p_open_home_median_nv"].to_numpy()

    for split, p_mkt in {
        "train": p_mkt_train,
        "val": p_mkt_val,
        "test": p_mkt_test,
    }.items():
        if ((p_mkt < 0) | (p_mkt > 1)).any():
            raise ValueError(
                f"p_open_home_median_nv contains values outside [0, 1] in the {split} split. "
                "Rebuild the MLP preprocessing cache with --force-recreate-preprocessing."
            )

    base_train = logit(torch.tensor(p_mkt_train, dtype=torch.float32)).unsqueeze(1)
    base_val = logit(torch.tensor(p_mkt_val, dtype=torch.float32)).unsqueeze(1)
    base_test = logit(torch.tensor(p_mkt_test, dtype=torch.float32)).unsqueeze(1)

    model_data['X_train'].drop(columns=["p_open_home_median_nv"], inplace=True)
    model_data['X_val'].drop(columns=["p_open_home_median_nv"], inplace=True)
    model_data['X_test'].drop(columns=["p_open_home_median_nv"], inplace=True)

    train_ds = TensorDataset(to_tensor(model_data['X_train']), base_train, to_tensor(model_data['y_train']))
    val_ds = TensorDataset(to_tensor(model_data['X_val']), base_val, to_tensor(model_data['y_val']))
    test_ds = TensorDataset(to_tensor(model_data['X_test']), base_test, to_tensor(model_data['y_test']))

    train_dl = DataLoader(train_ds, batch_size=config.train_batch, shuffle=False, pin_memory=False, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=config.val_batch, shuffle=False, pin_memory=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=config.val_batch, shuffle=False, pin_memory=False, num_workers=2)

    return {
        'in_dim': model_data["X_train"].shape[1],
        'train_dl': train_dl,
        'val_dl': val_dl,
        'test_dl': test_dl
    }

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim: int, base_hidden_size: int = 256, p_drop: float = 0.2, max_residual: float = 0.5):
        super().__init__()
        self.max_residual = max_residual

        self.net = nn.Sequential(
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

            nn.Linear(base_hidden_size, base_hidden_size // 2),
            nn.Dropout(p_drop),
            nn.ReLU(),
            
            nn.Linear(base_hidden_size // 2, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        raw_adjustment = self.net(x)
        mkt_adjustment = self.max_residual * torch.tanh(raw_adjustment)
        return mkt_adjustment    

class NNWrapper():
    def __init__(
            self, 
            config: NeuralNetworkConfig,
            in_dim: int, 
            train_dl: DataLoader, 
            val_dl: DataLoader, 
            test_dl: DataLoader):
        if not isinstance(config, NeuralNetworkConfig):
            raise TypeError("NNWrapper requires a NeuralNetworkConfig")
        
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.epochs = config.epochs
        self.max_residual = config.max_residual
        self.alpha = config.alpha
        self.base_hidden_size = config.base_hidden_size
        self.p_drop = config.p_drop
        self.config = config
        self.model_args = config
        self.hyperparam_path = HYPERPARAM_DIR / "nn_hyperparams.json"
        
        self.in_dim = in_dim
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
    def train_loop(self) -> float:
        size = len(self.train_dl.dataset)
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dl)

        for batch, (X, base_logit, y) in enumerate(self.train_dl):
            X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)

            mkt_adjustment = self.model(X)
            pred = base_logit + self.alpha * mkt_adjustment

            loss = self.loss_fn(pred, y)
            residual_loss = 0.1 * mkt_adjustment.pow(2).mean()
            loss += residual_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / num_batches
        return avg_train_loss

    def val_loop(self, dataloader) -> float:
        self.model.eval()
        num_batches = len(dataloader)
        val_loss = 0

        with torch.no_grad():
            for X, base_logit, y in dataloader:
                X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)
                mkt_adjustment = self.model(X)
                pred = base_logit + self.alpha * mkt_adjustment
                val_loss += self.loss_fn(pred, y).item()
                residual_loss = 0.4 * mkt_adjustment.pow(2).mean()
                val_loss += residual_loss.item()

        val_loss /= num_batches
        return val_loss
        

    def predict_proba(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0
        all_preds = []

        with torch.no_grad():
            for X, base_logit, y in dataloader:
                X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)
                
                mkt_adjustment = self.model(X)
                pred = base_logit + self.alpha * mkt_adjustment

                all_preds.extend(torch.sigmoid(pred.squeeze().detach().cpu()).numpy())
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"test Error: Avg loss: {test_loss:>8f} \n")
        return all_preds
    
    def train_and_eval_model(self, optimizer = None, loss_fn = None):
        retune = self.model_args.retune

        if loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()

        params = {}

        if retune:
            params = self.tune_hyperparameters()
            self._save_hyperparams(params)

        if self.model_args.use_hyperparams:
            print(f" Loading existing hyperparameters")
            try:
                params = self._load_hyperparams()
            except:
                raise RuntimeError("No hyperparameter Found!")
                
        if params:
            self.model = self._build_model_from_hparams(params).to(self.device)
            optimizer_name = params.get("optimizer", "AdamW")
            lr = params.get("lr", self.model_args.lr)
            weight_decay = params.get("weight_decay", 0.03)
            self.optimizer = getattr(optim, optimizer_name)(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.model = NeuralNetwork(
                            self.in_dim, 
                            base_hidden_size=self.base_hidden_size,
                            p_drop=self.p_drop,
                            max_residual=self.max_residual).to(self.device)
            
            if optimizer is None:
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    weight_decay=0.03,
                    lr=self.model_args.lr,
                )
            else:
                self.optimizer = optimizer

        self.scheduler = None
        if self.model_args.cosine_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.model_args.min_lr,
            )

        train_losses = []
        val_losses = []

        for e in range(self.epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            train_loss = self.train_loop()
            val_loss = self.val_loop(self.val_dl)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"train loss: {train_loss}")
            print(f"val loss: {val_loss}")
            if self.scheduler is not None:
                self.scheduler.step()
                print(f"lr: {self.optimizer.param_groups[0]['lr']:.2e}")

        self._plot_loss(train_losses, val_losses)
        self._save_model()
        
        return self.model
    
    def tune_hyperparameters(self):
        pbar = tqdm(total=50, desc="Optuna HPO")

        def tqdm_callback(study, trial):
            pbar.update(1)

        def define_model(trial):
            n_layers = trial.suggest_int("n_layers", 1, 4)
            layers = []
            
            in_feats = self.in_dim
            for i in range(n_layers):
                out_layer = trial.suggest_int("n_units_l{}".format(i), 32, 128)
                layers.append(nn.Linear(in_feats, out_layer))
                layers.append(nn.ReLU())
                dropout_layer = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(dropout_layer))
                in_feats = out_layer

            layers.append(nn.Linear(in_feats, 1))

            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

            return nn.Sequential(*layers)

        def objective(trial):
            model = define_model(trial).to(self.device)

            # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)  

            for epoch in range(350):
                model.train()

                for batch, (X, base_logit, y) in enumerate(self.train_dl):
                    X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)

                    mkt_adjustment = model(X)
                    pred = base_logit + mkt_adjustment

                    loss = self.loss_fn(pred, y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                model.eval()
                val_loss = 0

                with torch.no_grad():
                    for X, base_logit, y in self.val_dl:
                        X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)
                        mkt_adjustment = model(X)
                        pred = base_logit + mkt_adjustment
                        val_loss += self.loss_fn(pred, y).item()

                val_loss /= len(self.val_dl)
                trial.report(val_loss, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
            return val_loss
                
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, callbacks=[tqdm_callback])
        pbar.close()

        print(f"Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        print(f"Best Trial: {trial.value}")
        print(f"Best params:  {trial.params.items()}")

        return study.best_params

    def _plot_loss(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot training and validation loss curves across epochs.
        
        Args:
            train_losses: List of average training losses per epoch
            val_losses: List of validation losses per epoch
            plot_live: Display the loss curves live instead of saving a figure
            plot_interval: Epoch interval for live plot updates
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
        
        plot_path = PLOTS_DIR / "loss_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f" Loss curves saved to {plot_path}")

    def _build_model_from_hparams(self, hparams: Dict) -> nn.Module:
        """Build model architecture from hyperparameters"""
        n_layers = hparams.get("n_layers", 3)
        layers = []
        
        in_feats = self.in_dim
        for i in range(n_layers):
            out_layer = hparams.get(f"n_units_l{i}", 128)
            layers.append(nn.Linear(in_feats, out_layer))
            layers.append(nn.ReLU())
            # dropout = hparams.get(f"dropout_l{i}", 0.2)
            # layers.append(nn.Dropout(dropout))
            in_feats = out_layer

        layers.append(nn.Linear(in_feats, 1))
        
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        return nn.Sequential(*layers)
    
    def _save_hyperparams(self, params: Dict) -> None:
        with open(self.hyperparam_path, 'w') as f:
            json.dump(params, f)

    def _load_hyperparams(self) -> Dict:
        if self.hyperparam_path.exists():
            return json.loads(self.hyperparam_path.read_text())
        
        print(f" No hyperparameters found at {self.hyperparam_path}")
        return {}

    def _save_model(self) -> None:
        model_path = SAVED_MODEL_DIR / "saved_nn.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }, model_path)
        print(f" Successfully saved Neural Network model to {model_path}")

    def load_model(self):
        model_path = SAVED_MODEL_DIR / "saved_nn.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f" Successfully loaded Neural Network model")
            return self.model
        
        print(f" No Neural Network model found")
        return None

if __name__ == "__main__":
    args = create_args()
    config = NeuralNetworkConfig.from_namespace(
        argparse.Namespace(**vars(args), stage="finetune")
    )

    data = prepare_mlp_data(
        config=config,
    )
    
    mlp = NNWrapper(
        config=config,
        in_dim=data['in_dim'], 
        train_dl=data['train_dl'], 
        val_dl=data['val_dl'], 
        test_dl=data['test_dl'],
    )
    mlp.train_and_eval_model()

    # preds = mlp.predict_proba(mlp.test_dl)

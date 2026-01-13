from pandas.core.api import DataFrame as DataFrame
import pandas as pd
from typing import Dict
import optuna, json, argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT
from src.utils import setup_logging

HYPERPARAM_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_hyperparameters"
HYPERPARAM_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODEL_DIR = PROJECT_ROOT / "src" / "data" / "models" / "saved_models"
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "src" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "neural_network_log.log"

def create_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Network Model Pipeline")
    parser.add_argument("--retune", action='store_true', help='Retune hyperparameters')
    parser.add_argument("--force-recreate", action="store_true", help="Recreate rolling features, even if cached file exists")
    parser.add_argument("--force-recreate-preprocessing", action="store_true", help="Recreate preprocessed datasets, even if cached file exists")
    parser.add_argument("--log", action="store_true", help=f"Write debug data to log file {LOG_FILE}")
    parser.add_argument("--log-file", type=str, help="Custom log file path (overrides default)")
    parser.add_argument("--clear-log", action="store_true", help="Clear the log file before starting (removes existing log content)")
    return parser.parse_args()

def prepare_mlp_data(model_data: Dict[str, DataFrame] | None = None, force_recreate: bool = False, force_recreate_preprocessing: bool = False) -> Dict[str, int | DataLoader]:
    def to_tensor(x: DataFrame) -> torch.Tensor:
        return torch.tensor(x.values, dtype=torch.float32)
    
    def logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)
    
    if not model_data:
        model_data, _ = PreProcessing([2021, 2022, 2023, 2024, 2025], model_type='mlp', mkt_only=False).preprocess_feats(
                    force_recreate=force_recreate,
                    force_recreate_preprocessing=force_recreate_preprocessing,
            )

    p_mkt_train = model_data["X_train"]["p_open_home_median_nv"].to_numpy()
    p_mkt_test = model_data["X_test"]["p_open_home_median_nv"].to_numpy()
    p_mkt_val = model_data["X_val"]["p_open_home_median_nv"].to_numpy()

    base_train = logit(torch.tensor(p_mkt_train, dtype=torch.float32)).unsqueeze(1)
    base_val = logit(torch.tensor(p_mkt_val, dtype=torch.float32)).unsqueeze(1)
    base_test = logit(torch.tensor(p_mkt_test, dtype=torch.float32)).unsqueeze(1)

    model_data['X_train'].drop(columns=["p_open_home_median_nv"], inplace=True)
    model_data['X_val'].drop(columns=["p_open_home_median_nv"], inplace=True)
    model_data['X_test'].drop(columns=["p_open_home_median_nv"], inplace=True)

    train_ds = TensorDataset(to_tensor(model_data['X_train']), base_train, to_tensor(model_data['y_train']))
    val_ds = TensorDataset(to_tensor(model_data['X_val']), base_val, to_tensor(model_data['y_val']))
    test_ds = TensorDataset(to_tensor(model_data['X_test']), base_test, to_tensor(model_data['y_test']))

    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=False, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=8192, shuffle=False, pin_memory=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=8192, shuffle=False, pin_memory=False, num_workers=2)

    return {
        'in_dim': model_data["X_train"].shape[1],
        'train_dl': train_dl,
        'val_dl': val_dl,
        'test_dl': test_dl
    }


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


    def forward(self, x):
        mkt_adjustment = self.net(x)
        return mkt_adjustment    

class NNWrapper():
    def __init__(self, model_args, in_dim: int, train_dl: DataLoader, val_dl: DataLoader, test_dl: DataLoader, epochs: int = 100):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.epochs = epochs
        self.model_args = model_args
        self.hyperparam_path = HYPERPARAM_DIR / "nn_hyperparams.json"
        
        self.in_dim = in_dim
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
    def train_loop(self):
        size = len(self.train_dl.dataset)
        self.model.train()

        for batch, (X, base_logit, y) in enumerate(self.train_dl):
            X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)

            mkt_adjustment = self.model(X)
            pred = base_logit + mkt_adjustment

            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def val_loop(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        val_loss = 0

        with torch.no_grad():
            for X, base_logit, y in dataloader:
                X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)
                mkt_adjustment = self.model(X)
                pred = base_logit + mkt_adjustment
                val_loss += self.loss_fn(pred, y).item()

        val_loss /= num_batches
        print(f"Val Error: Avg loss: {val_loss:>8f} \n")
        

    def predict_proba(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0
        all_preds = []

        with torch.no_grad():
            for X, base_logit, y in dataloader:
                X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)
                
                mkt_adjustment = self.model(X)
                pred = base_logit + mkt_adjustment

                all_preds.extend(torch.sigmoid(pred.squeeze().detach().cpu()).numpy())
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"test Error: Avg loss: {test_loss:>8f} \n")
        return all_preds
    
    def train_and_eval_model(self, optimizer = None, loss_fn = None):
        retune = model_args.retune

        if loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss()

        if retune:
            params = self.tune_hyperparameters()
            self._save_hyperparams(params)
        else:
            print(f" Loading existing hyperparameters")
            try:
                params = self._load_hyperparams()
            except:
                params = {}
                
        if params:
            self.model = self._build_model_from_hparams(params).to(self.device)
            optimizer_name = params.get("optimizer", "AdamW")
            lr = params.get("lr", 1e-3)
            self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr)
        else:
            self.model = NeuralNetwork(self.in_dim).to(self.device)
            if optimizer is None:
                self.optimizer = optim.AdamW(self.model.parameters())

        for e in range(self.epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            self.train_loop()
            self.val_loop(self.val_dl)
        
        self._save_model()
        
        return self.model
    
    def tune_hyperparameters(self):
        pbar = tqdm(total=200, desc="Optuna HPO")

        def tqdm_callback(study, trial):
            pbar.update(1)

        def define_model(trial):
            n_layers = trial.suggest_int("n_layers", 1, 5)
            layers = []
            
            in_feats = self.in_dim
            for i in range(n_layers):
                out_layer = trial.suggest_int("n_units_l{}".format(i), 64, 512)
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

            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)  

            for epoch in range(100):
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
        study.optimize(objective, n_trials=200, callbacks=[tqdm_callback])
        pbar.close()

        print(f"Number of finished trials: {len(study.trials)}")

        trial = study.best_trial
        print(f"Best Trial: {trial.value}")
        print(f"Best params:  {trial.params.items()}")

        return study.best_params

    def _build_model_from_hparams(self, hparams: Dict) -> nn.Module:
        """Build model architecture from hyperparameters"""
        n_layers = hparams.get("n_layers", 3)
        layers = []
        
        in_feats = self.in_dim
        for i in range(n_layers):
            out_layer = hparams.get(f"n_units_l{i}", 128)
            layers.append(nn.Linear(in_feats, out_layer))
            layers.append(nn.ReLU())
            dropout = hparams.get(f"dropout_l{i}", 0.2)
            layers.append(nn.Dropout(dropout))
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
        }, model_path)
        print(f" Successfully saved Neural Network model to {model_path}")

    def load_model(self):
        model_path = SAVED_MODEL_DIR / "saved_nn.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f" Successfully loaded Neural Network model")
            return self.model
        
        print(f" No Neural Network model found")
        return None

if __name__ == "__main__":
    model_args = create_args()

    data = prepare_mlp_data(model_args.force_recreate, model_args.force_recreate_preprocessing)
    
    mlp = NNWrapper(model_args=model_args, in_dim=data['in_dim'], train_dl=data['train_dl'], val_dl=data['val_dl'], test_dl=data['test_dl'])
    mlp.train_and_eval_model()

    preds = mlp.predict_proba(mlp.test_dl)



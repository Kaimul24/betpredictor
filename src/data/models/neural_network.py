from pandas.core.api import DataFrame as DataFrame
import pandas as pd
from typing import Dict
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.models.calibration import select_and_fit_calibrator, save_calibrator, load_calibrator, apply_calibration, plot_calibration
from src.data.feature_preprocessing import PreProcessing
from src.config import PROJECT_ROOT
from src.utils import setup_logging

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


    def forward(self, x, mkt_logit):
        mkt_adjustment = self.net(x)
        updated_odds = mkt_logit + mkt_adjustment
        return updated_odds    

class NNWrapper():
    def __init__(self, epochs: int = 100, optimizer = None, loss_fn = None):
        # self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.device = 'cpu'
        self.epochs = epochs
        self.prepare_mlp_data()
        self.model = NeuralNetwork(self.in_dim).to(self.device)

        if optimizer is None:
            self.optimizer = optim.AdamW(self.model.parameters())

        if loss_fn is None:
            self.loss_fn = nn.BCEWithLogitsLoss() 

    def prepare_mlp_data(self) -> None:

        def to_tensor(x: DataFrame) -> torch.Tensor:
            return torch.tensor(x.values, dtype=torch.float32)
        
        def logit(p, eps=1e-6):
            p = torch.clamp(p, eps, 1 - eps)
            return torch.log(p) - torch.log1p(-p)

        model_data, _ = PreProcessing([2021, 2022, 2023, 2024, 2025], model_type='mlp', mkt_only=False ).preprocess_feats(
                force_recreate=True,
                force_recreate_preprocessing=True,
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

        self.in_dim = model_data['X_train'].shape[1]
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
    def train_loop(self):
        size = len(self.train_dl.dataset)
        self.model.train()

        for batch, (X, base_logit, y) in enumerate(self.train_dl):
            X, base_logit, y = X.to(self.device), base_logit.to(self.device), y.to(self.device)

            pred = self.model(X, base_logit)
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
                pred = self.model(X, base_logit)
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
                pred = self.model(X, base_logit)
                all_preds.extend(torch.sigmoid(pred.squeeze().detach().cpu()).numpy())
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"test Error: Avg loss: {test_loss:>8f} \n")
        return all_preds
    
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

            ## NEED A WAY TO CALL FORWARD TO ADD THE MARKET PRIORS TO THE MODEL CALL


        
    

if __name__ == "__main__":
    mlp = NNWrapper()

    for e in range(100):
        print(f"Epoch {e+1}\n-------------------------------")
        mlp.train_loop()
        mlp.val_loop(mlp.val_dl)

    preds = mlp.predict_proba(mlp.test_dl)
    # print(preds)


        
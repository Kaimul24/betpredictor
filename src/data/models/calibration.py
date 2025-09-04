import json
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

def plot_calibration(y_true, y_proba, split: str,  filepath: str,  n_bins: int = 10) -> str:
        """Plot and save reliability (calibration) curve with probability histogram."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
        plt.figure(figsize=(6, 6))
        
        plt.plot(prob_pred, prob_true, marker='o', label='Model', linewidth=1.2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"Calibration Curve ({split})")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)

        out_path_curve = f"{filepath}_{split}.png"
        plt.tight_layout()
        plt.savefig(out_path_curve, dpi=120)
        plt.close()
        return out_path_curve
        


def _clip_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Clip probabilities to (eps, 1-eps) for numerical stability (logloss, logits)."""
    return np.clip(p, eps, 1.0 - eps)


def _to_logit(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Convert probabilities to logits (safe with clipping)."""
    p = _clip_probs(np.asarray(p, dtype=float), eps=eps)
    return np.log(p / (1.0 - p))


def _bin_edges(y_prob: np.ndarray, n_bins: int, strategy: Literal["uniform", "quantile"]) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, qs)
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def _bin_stats_from_edges(y_true: np.ndarray, y_prob: np.ndarray, edges: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prob_true, prob_pred, counts) for fixed bin edges."""
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    n_bins = len(edges) - 1
    prob_true = np.empty(n_bins, float)
    prob_pred = np.empty(n_bins, float)
    counts = np.empty(n_bins, int)

    for b in range(n_bins):
        m = (idx == b)
        counts[b] = int(m.sum())
        if counts[b] > 0:
            prob_true[b] = float(np.mean(y_true[m]))
            prob_pred[b] = float(np.mean(y_prob[m]))
        else:
            prob_true[b] = np.nan
            prob_pred[b] = float((edges[b] + edges[b + 1]) / 2.0)
    return prob_true, prob_pred, counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """
    ECE = sum_k (n_k / N) * |acc_k - conf_k| over non-empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = _bin_edges(y_prob, n_bins, strategy)
    prob_true, prob_pred, counts = _bin_stats_from_edges(y_true, y_prob, edges)
    m = counts > 0
    if not np.any(m):
        return float("nan")
    weights = counts[m] / counts[m].sum()
    return float(np.sum(weights * np.abs(prob_true[m] - prob_pred[m])))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(log_loss(y_true, _clip_probs(y_prob)))


@dataclass
class BaseCalibrator:
    kind: Literal["platt", "isotonic"]

    def transform(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(obj: Dict) -> "BaseCalibrator":
        kind = obj.get("kind")
        if kind == "platt":
            return PlattCalibrator(a=float(obj["a"]), b=float(obj["b"]))
        if kind == "isotonic":
            x = np.asarray(obj["x"], dtype=float)
            y = np.asarray(obj["y"], dtype=float)
            return IsotonicCalibrator(x_thresholds=x, y_thresholds=y)
        raise ValueError(f"Unknown calibrator kind: {kind}")


@dataclass
class PlattCalibrator(BaseCalibrator):
    """
    Platt scaling: sigmoid(a * logit(p) + b).
    Fit is just a logistic regression on logit(p) -> y.
    """
    a: float
    b: float

    def __init__(self, a: float = 1.0, b: float = 0.0):
        super().__init__(kind="platt")
        self.a = a
        self.b = b

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = self.a * _to_logit(p) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> Dict:
        return {"kind": "platt", "a": float(self.a), "b": float(self.b)}


@dataclass
class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression mapping stored as thresholds so we can serialize to JSON.
    transform() uses linear interpolation with clipping to [0,1].
    """
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def __init__(self, x_thresholds: np.ndarray, y_thresholds: np.ndarray):
        super().__init__(kind="isotonic")
        self.x_thresholds = np.asarray(x_thresholds, dtype=float)
        self.y_thresholds = np.asarray(y_thresholds, dtype=float)

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)

        out = np.interp(p, self.x_thresholds, self.y_thresholds)

        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            "kind": "isotonic",
            "x": self.x_thresholds.astype(float).tolist(),
            "y": self.y_thresholds.astype(float).tolist(),
        }

def fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> PlattCalibrator:
    """
    Fit Platt scaling by logistic regression on logit(p_cal).
    """
    y = np.asarray(y_cal, dtype=float).ravel()
    z = _to_logit(np.asarray(p_cal, dtype=float)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(z, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return PlattCalibrator(a=a, b=b)


def fit_isotonic(y_cal: np.ndarray, p_cal: np.ndarray) -> IsotonicCalibrator:
    """
    Fit isotonic regression (monotone mapping p_cal -> y).
    Serialize thresholds for lightweight artifact.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    y = np.asarray(y_cal, dtype=float).ravel()
    p = np.asarray(p_cal, dtype=float).ravel()
    iso.fit(p, y)
    return IsotonicCalibrator(x_thresholds=iso.X_thresholds_, y_thresholds=iso.y_thresholds_)


def select_and_fit_calibrator(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    *,
    methods: Iterable[Literal["platt", "isotonic"]] = ("platt", "isotonic"),
    selection_metric: Literal["ece", "brier", "logloss"] = "ece",
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
    y_eval: Optional[np.ndarray] = None,
    p_eval_raw: Optional[np.ndarray] = None,
) -> Tuple[BaseCalibrator, Dict[str, Dict[str, float]]]:
    """
    Fit candidate calibrators on (y_cal, p_cal), evaluate on either:
      - provided (y_eval, p_eval_raw) if given (preferred), else
      - the calibration set itself (risk of optimistic selection).
    Return (best_calibrator, per_method_metrics).

    per_method_metrics[method] includes {'ece','brier','logloss'}.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)

    fitted = {}
    if "platt" in methods:
        fitted["platt"] = fit_platt(y_cal, p_cal)
    if "isotonic" in methods:
        fitted["isotonic"] = fit_isotonic(y_cal, p_cal)

    if y_eval is not None and p_eval_raw is not None:
        y_eval = np.asarray(y_eval, dtype=float)
        p_eval_raw = np.asarray(p_eval_raw, dtype=float)
        eval_y = y_eval
        def eval_probs(cal: BaseCalibrator) -> np.ndarray:
            return cal.transform(p_eval_raw)
    else:
        eval_y = y_cal
        def eval_probs(cal: BaseCalibrator) -> np.ndarray:
            return cal.transform(p_cal)

    metrics = {}
    for name, cal in fitted.items():
        p_eval = eval_probs(cal)
        metrics[name] = {
            "ece": expected_calibration_error(eval_y, p_eval, n_bins=n_bins, strategy=strategy),
            "brier": brier(eval_y, p_eval),
            "logloss": nll(eval_y, p_eval),
        }

    key = selection_metric
    best_name = min(metrics.keys(), key=lambda k: metrics[k][key])
    return fitted[best_name], metrics


def apply_calibration(calibrator: BaseCalibrator, p_raw: np.ndarray) -> np.ndarray:
    """
    Apply a fitted calibrator to raw probabilities.
    """
    return calibrator.transform(p_raw)


def save_calibrator(calibrator: BaseCalibrator, path: str) -> None:
    """
    Save calibrator to JSON (portable, git-friendly).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f, indent=2)


def load_calibrator(path: str) -> BaseCalibrator:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return BaseCalibrator.from_dict(obj)

    
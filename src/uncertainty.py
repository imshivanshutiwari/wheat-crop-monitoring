"""Uncertainty quantification for yield forecasts.

- SplitConformal : distribution-free prediction intervals with finite-sample
  coverage guarantees (Vovk; Romano et al. 2019 CQR-style split).
- mc_dropout_predict : Monte-Carlo dropout predictive mean & std for the
  PyTorch deep models.
- ensemble_intervals : intervals from the EnKF / model-ensemble yields.
"""
import numpy as np


class SplitConformal:
    """Split conformal regression intervals around any point predictor.

    Guarantees marginal coverage >= 1 - alpha on exchangeable data.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None

    def calibrate(self, y_true, y_pred):
        residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        n = len(residuals)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = float(np.quantile(residuals, min(level, 1.0)))
        return self.q

    def interval(self, y_pred):
        if self.q is None:
            raise RuntimeError("Call calibrate() first.")
        y_pred = np.asarray(y_pred, float)
        return y_pred - self.q, y_pred + self.q


def mc_dropout_predict(model, x, n_samples=50):
    """Predictive mean/std via MC dropout (keeps dropout active at inference).

    model: torch.nn.Module with dropout layers; x: torch.Tensor.
    """
    import torch
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = torch.softmax(model(x), dim=-1)
            preds.append(out.cpu().numpy())
    preds = np.stack(preds)               # (S, N, C)
    return preds.mean(0), preds.std(0)


def ensemble_intervals(ensemble_values, alpha=0.1):
    """Empirical prediction interval from ensemble samples (e.g. EnKF yields)."""
    lo = np.quantile(ensemble_values, alpha / 2)
    hi = np.quantile(ensemble_values, 1 - alpha / 2)
    return float(ensemble_values.mean()), float(lo), float(hi)

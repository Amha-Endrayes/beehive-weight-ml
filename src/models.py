import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    class BaseEstimator:
        pass
    class RegressorMixin:
        pass
    class HistGradientBoostingRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self
        def predict(self, X):
            return np.full(len(X), getattr(self, 'mean_', 0.0))

class CalibratedBeehiveEstimator(BaseEstimator, RegressorMixin):
    """
    Two-stage hybrid estimator:
    1. Base Regressor: Predicts relative weight dynamic response from sensor inputs.
    2. Tare Offset Calibrator: Anchors static hardware tare from initial calibration window (e.g. 24h).
    """
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, random_state=42
        )
        self.tare_offsets = {}

    def fit(self, X_features, y_weight, colony_ids=None):
        if colony_ids is not None:
            df_y = pd.DataFrame({'y': y_weight, 'colony': colony_ids})
            self.tare_offsets = df_y.groupby('colony')['y'].mean().to_dict()
            y_centered = y_weight - df_y['colony'].map(self.tare_offsets)
            self.base_estimator.fit(X_features, y_centered)
        else:
            self.base_estimator.fit(X_features, y_weight)
        return self

    def predict(self, X_features, colony_ids=None, calibration_tare=None):
        base_preds = self.base_estimator.predict(X_features)
        if calibration_tare is not None:
            return base_preds + calibration_tare
        elif colony_ids is not None:
            offsets = np.array([self.tare_offsets.get(cid, np.mean(list(self.tare_offsets.values()) or [0.0])) 
                               for cid in colony_ids])
            return base_preds + offsets
        return base_preds

from __future__ import annotations

import numpy as np


class RobustScale:
    """Median / IQR scaler. Fitted on training rows only."""

    def __init__(self):
        self.center = None
        self.scale = None

    def fit(self, values: np.ndarray) -> "RobustScale":
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        finite = np.isfinite(values).all(axis=1)
        if not np.any(finite):
            raise ValueError("Cannot scale a column with no finite values")
        used = values[finite]
        self.center = np.median(used, axis=0)
        spread = np.percentile(used, 75, axis=0) - np.percentile(used, 25, axis=0)
        self.scale = np.where(spread < 1e-6, 1.0, spread)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        out = (values - self.center) / self.scale
        return out[:, 0] if squeeze else out

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        out = values * self.scale + self.center
        return out[:, 0] if squeeze else out

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "RobustScale":
        scaler = cls()
        scaler.center = np.asarray(payload["center"], dtype=float)
        scaler.scale = np.asarray(payload["scale"], dtype=float)
        return scaler

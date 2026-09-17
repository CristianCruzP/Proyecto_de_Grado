from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import PREPROCESSORS_DIR, ensure_directories


def chronological_split(df: pd.DataFrame, date_col: str = "month", train_size: float = 0.7, val_size: float = 0.15):
    data = df.sort_values(date_col).reset_index(drop=True)
    n = len(data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    return data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy()


def fit_train_scaler(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[list(feature_cols)])
    return scaler


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out[list(feature_cols)] = scaler.transform(out[list(feature_cols)])
    return out


def save_scaler(scaler: StandardScaler, file_name: str = "lstm_scaler.joblib") -> Path:
    ensure_directories()
    PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREPROCESSORS_DIR / file_name
    joblib.dump(scaler, path)
    return path


def build_lstm_windows(features: np.ndarray, target: np.ndarray, dates: np.ndarray, window_size: int, horizon: int = 1):
    X, y, y_dates = [], [], []
    for idx in range(window_size, len(features) - horizon + 1):
        X.append(features[idx - window_size : idx])
        y.append(target[idx + horizon - 1])
        y_dates.append(dates[idx + horizon - 1])
    if not X:
        return np.empty((0, window_size, features.shape[1])), np.empty((0,)), np.array([])
    return np.array(X), np.array(y), np.array(y_dates)


def prepare_lstm_datasets(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: str = "month",
    window_size: int = 12,
    horizon: int = 1,
    train_size: float = 0.7,
    val_size: float = 0.15,
):
    ordered = df.sort_values(date_col).reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(ordered, date_col=date_col, train_size=train_size, val_size=val_size)

    scaler = fit_train_scaler(train_df.dropna(subset=list(feature_cols)), feature_cols)
    scaled = ordered.copy()
    scaled[list(feature_cols)] = scaler.transform(scaled[list(feature_cols)])

    X, y, y_dates = build_lstm_windows(
        scaled[list(feature_cols)].to_numpy(),
        scaled[target_col].to_numpy(),
        scaled[date_col].to_numpy(),
        window_size=window_size,
        horizon=horizon,
    )

    train_end = len(train_df)
    val_end = len(train_df) + len(val_df)

    train_mask = y_dates < ordered[date_col].iloc[train_end] if train_end < len(ordered) else np.ones_like(y_dates, dtype=bool)
    val_mask = (y_dates >= ordered[date_col].iloc[train_end]) & (y_dates < ordered[date_col].iloc[val_end]) if val_end < len(ordered) and train_end < len(ordered) else np.zeros_like(y_dates, dtype=bool)
    test_mask = ~(train_mask | val_mask)

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "target_dates_train": y_dates[train_mask],
        "target_dates_val": y_dates[val_mask],
        "target_dates_test": y_dates[test_mask],
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }

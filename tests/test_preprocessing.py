import numpy as np
import pandas as pd

from src.preprocessing import chronological_split, prepare_lstm_datasets


def test_chronological_split_preserves_order():
    df = pd.DataFrame({"month": pd.date_range("2000-01-01", periods=10, freq="MS"), "x": range(10)})
    train, val, test = chronological_split(df, "month", train_size=0.6, val_size=0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train["month"].max() < val["month"].min() < test["month"].min()


def test_prepare_lstm_datasets_scaler_and_windows():
    df = pd.DataFrame(
        {
            "month": pd.date_range("2000-01-01", periods=24, freq="MS"),
            "f1": np.linspace(0, 23, 24),
            "f2": np.linspace(10, 33, 24),
            "target": np.linspace(100, 123, 24),
        }
    )
    datasets = prepare_lstm_datasets(
        df,
        feature_cols=["f1", "f2"],
        target_col="target",
        window_size=6,
        train_size=0.5,
        val_size=0.25,
    )

    assert datasets["X_train"].shape[1:] == (6, 2)
    assert len(datasets["X_train"]) > 0
    assert len(datasets["X_test"]) > 0
    assert np.all(np.diff(datasets["target_dates_train"].astype("datetime64[M]")) >= np.timedelta64(0, "M"))

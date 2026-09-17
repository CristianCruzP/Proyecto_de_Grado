from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def is_tensorflow_available() -> bool:
    try:
        import tensorflow as _tf  # noqa: F401

        return True
    except Exception:
        return False


def build_lstm_model(input_shape, units: int = 32, dropout: float = 0.2, learning_rate: float = 1e-3):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def _metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
    }


def train_lstm_or_fallback(datasets: dict, epochs: int = 30, batch_size: int = 32):
    X_train, y_train = datasets["X_train"], datasets["y_train"]
    X_val, y_val = datasets["X_val"], datasets["y_val"]
    X_test, y_test = datasets["X_test"], datasets["y_test"]

    if is_tensorflow_available() and len(X_train):
        import tensorflow as tf

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val) if len(X_val) else None, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
        pred_val = model.predict(X_val, verbose=0).ravel() if len(X_val) else np.array([])
        pred_test = model.predict(X_test, verbose=0).ravel() if len(X_test) else np.array([])
        return {
            "model_type": "tensorflow_lstm",
            "model": model,
            "validation": _metrics(y_val, pred_val),
            "test": _metrics(y_test, pred_test),
        }

    reg = DummyRegressor(strategy="mean")
    X_train_flat = X_train.reshape((X_train.shape[0], -1)) if len(X_train) else np.empty((0, 0))
    X_val_flat = X_val.reshape((X_val.shape[0], -1)) if len(X_val) else np.empty((0, 0))
    X_test_flat = X_test.reshape((X_test.shape[0], -1)) if len(X_test) else np.empty((0, 0))
    if len(X_train_flat):
        reg.fit(X_train_flat, y_train)
    pred_val = reg.predict(X_val_flat) if len(X_val_flat) else np.array([])
    pred_test = reg.predict(X_test_flat) if len(X_test_flat) else np.array([])

    return {
        "model_type": "dummy_fallback",
        "model": reg,
        "validation": _metrics(y_val, pred_val),
        "test": _metrics(y_test, pred_test),
    }

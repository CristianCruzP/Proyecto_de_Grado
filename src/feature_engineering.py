import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from .config import GOLD_DIR, SILVER_DIR, ensure_directories


def compute_spi_gamma(precip: pd.Series, scale: int = 3, min_non_null: int = 30) -> pd.Series:
    rolling = precip.rolling(window=scale, min_periods=scale).sum()
    valid = rolling.dropna()
    if valid.shape[0] < min_non_null:
        return pd.Series(np.nan, index=precip.index)

    positive = valid[valid > 0]
    if positive.empty:
        return pd.Series(np.nan, index=precip.index)

    shape, loc, sc = gamma.fit(positive, floc=0)
    q0 = (valid == 0).mean()
    cdf = pd.Series(index=rolling.index, dtype=float)
    cdf.loc[rolling.notna()] = q0 + (1 - q0) * gamma.cdf(rolling[rolling.notna()].clip(lower=0), shape, loc=loc, scale=sc)
    cdf = cdf.clip(1e-8, 1 - 1e-8)
    spi = pd.Series(norm.ppf(cdf), index=rolling.index)
    return spi


def build_agriculture_auxiliary(silver_dir: Path, gold_dir: Path) -> pd.DataFrame:
    eva_path = silver_dir / "agricultura_eva_normalizada.csv"
    eva = pd.read_csv(eva_path)
    eva["year"] = pd.to_numeric(eva["year"], errors="coerce")
    aux = (
        eva.dropna(subset=["year", "dane_municipio"])
        .groupby(["year", "dane_municipio", "municipio", "departamento"], as_index=False)
        .agg(
            area_sembrada_ha=("area_sembrada_ha", "sum"),
            area_cosechada_ha=("area_cosechada_ha", "sum"),
            produccion_t=("produccion_t", "sum"),
            rendimiento_promedio_t_ha=("rendimiento_t_ha", "mean"),
        )
    )
    aux["integration_note"] = "Serie agricola se conserva como tabla auxiliar anual; no se fuerza union mensual causal con objetivo LSTM"
    aux.to_csv(gold_dir / "agricultura_eva_anual_auxiliar.csv", index=False)
    return aux


def build_gold_dataset(
    silver_dir: Path = SILVER_DIR,
    gold_dir: Path = GOLD_DIR,
    spi_scale: int = 3,
    oni_lags: Sequence[int] = (1, 3, 6),
    drought_threshold: float = -1.0,
) -> pd.DataFrame:
    ensure_directories()
    gold_dir.mkdir(parents=True, exist_ok=True)

    precip = pd.read_csv(silver_dir / "precipitacion_chicoral_mensual.csv", parse_dates=["month"])
    temp = pd.read_csv(silver_dir / "temperatura_chicoral_mensual.csv", parse_dates=["month"])
    oni = pd.read_csv(silver_dir / "oni_mensual.csv", parse_dates=["month"])

    climate = precip.merge(temp, on="month", how="outer").merge(oni[["month", "oni_anomaly"]], on="month", how="left")
    climate = climate.sort_values("month").drop_duplicates(subset=["month"], keep="first").reset_index(drop=True)

    climate["spi_gamma"] = compute_spi_gamma(climate["precip_mm"], scale=spi_scale)
    climate["year"] = climate["month"].dt.year
    climate["month_num"] = climate["month"].dt.month
    climate["sin_month"] = np.sin(2 * np.pi * climate["month_num"] / 12)
    climate["cos_month"] = np.cos(2 * np.pi * climate["month_num"] / 12)

    for lag in oni_lags:
        climate[f"oni_lag_{lag}"] = climate["oni_anomaly"].shift(lag)

    drought = pd.Series(np.nan, index=climate.index)
    valid = climate["spi_gamma"].notna()
    drought.loc[valid] = (climate.loc[valid, "spi_gamma"] <= drought_threshold).astype(int)
    climate["drought_label"] = drought

    climate.to_csv(gold_dir / "clima_sequia_mensual.csv", index=False)
    climate.to_csv(gold_dir / "dataset_lstm_mensual.csv", index=False)

    n = len(climate)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    split_df = pd.DataFrame(
        [
            {"split": "train", "start": climate["month"].iloc[0] if n else None, "end": climate["month"].iloc[train_end - 1] if train_end else None},
            {"split": "validation", "start": climate["month"].iloc[train_end] if n > train_end else None, "end": climate["month"].iloc[val_end - 1] if val_end > train_end else None},
            {"split": "test", "start": climate["month"].iloc[val_end] if n > val_end else None, "end": climate["month"].iloc[-1] if n else None},
        ]
    )
    split_df.to_csv(gold_dir / "train_validation_test_dates.csv", index=False)

    metadata = {
        "rows": int(n),
        "columns": list(climate.columns),
        "spi_scale": spi_scale,
        "oni_lags": list(oni_lags),
        "drought_threshold": drought_threshold,
        "null_counts": {k: int(v) for k, v in climate.isna().sum().to_dict().items()},
        "notes": [
            "No se imputan faltantes automaticamente; NaN se preservan para trazabilidad.",
            "La capa agricola EVA se conserva como tabla auxiliar anual por cobertura no mensual.",
        ],
    }
    with open(gold_dir / "dataset_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, default=str)

    build_agriculture_auxiliary(silver_dir, gold_dir)
    return climate

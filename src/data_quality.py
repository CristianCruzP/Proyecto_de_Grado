from pathlib import Path

import pandas as pd

from .config import DATA_QUALITY_DIR, SILVER_DIR, ensure_directories


def summarize_dataset(df: pd.DataFrame, name: str, date_col: str | None = None, key_cols: list[str] | None = None) -> pd.DataFrame:
    row = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "null_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if key_cols:
        existing = [c for c in key_cols if c in df.columns]
        row["duplicate_keys"] = int(df.duplicated(subset=existing).sum()) if existing else None
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        row["date_min"] = dates.min()
        row["date_max"] = dates.max()
        row["date_nulls"] = int(dates.isna().sum())
    return pd.DataFrame([row])


def monthly_continuity(df: pd.DataFrame, date_col: str, dataset: str) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().drop_duplicates()
    if dates.empty:
        return pd.DataFrame(columns=["dataset", "month", "is_missing"])
    full = pd.date_range(dates.min(), dates.max(), freq="MS")
    missing = full.difference(dates)
    if len(missing) == 0:
        return pd.DataFrame([{"dataset": dataset, "month": None, "is_missing": False}])
    return pd.DataFrame({"dataset": dataset, "month": missing, "is_missing": True})


def run_silver_quality_checks(silver_dir: Path = SILVER_DIR, report_dir: Path = DATA_QUALITY_DIR) -> dict[str, pd.DataFrame]:
    ensure_directories()
    report_dir.mkdir(parents=True, exist_ok=True)

    precip = pd.read_csv(silver_dir / "precipitacion_chicoral_mensual.csv")
    temp = pd.read_csv(silver_dir / "temperatura_chicoral_mensual.csv")
    oni = pd.read_csv(silver_dir / "oni_mensual.csv")
    eva = pd.read_csv(silver_dir / "agricultura_eva_normalizada.csv")

    quality = pd.concat(
        [
            summarize_dataset(precip, "precipitacion", "month", ["month"]),
            summarize_dataset(temp, "temperatura", "month", ["month"]),
            summarize_dataset(oni, "oni", "month", ["month"]),
            summarize_dataset(eva, "agricultura", "year_start", ["dane_municipio", "year", "periodo", "cultivo"]),
        ],
        ignore_index=True,
    )

    continuity = pd.concat(
        [
            monthly_continuity(precip, "month", "precipitacion"),
            monthly_continuity(temp, "month", "temperatura"),
            monthly_continuity(oni, "month", "oni"),
        ],
        ignore_index=True,
    )

    quality.to_csv(silver_dir / "quality_report.csv", index=False)
    quality.to_csv(report_dir / "quality_report.csv", index=False)
    continuity.to_csv(report_dir / "monthly_continuity_report.csv", index=False)

    return {"quality": quality, "continuity": continuity}

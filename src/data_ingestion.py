import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import BRONZE_DIR, DATA_QUALITY_DIR, SILVER_DIR, ensure_directories


ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[
]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_columns(columns: Iterable[str]) -> list[str]:
    normalized = []
    for col in columns:
        clean = _normalize_text(col)
        clean = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
        normalized.append(clean)
    return normalized


def parse_localized_number(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    text = text.replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_year(value) -> float:
    number = parse_localized_number(value)
    if np.isnan(number):
        return np.nan
    if number > 1900 and number < 3000:
        return int(number)
    maybe = re.sub(r"\D", "", str(value))
    if len(maybe) >= 4:
        return int(maybe[:4])
    return np.nan


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover - fallback branch
            last_error = exc
    raise ValueError(f"No se pudo leer {path}") from last_error


def build_bronze_manifest(bronze_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    rows = []
    for file_path in sorted(bronze_dir.rglob("*.csv")):
        rel = file_path.relative_to(bronze_dir)
        raw = file_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        encoding = "unknown"
        for enc in ENCODINGS:
            try:
                raw.decode(enc)
                encoding = enc
                break
            except Exception:
                continue
        df = read_csv_flexible(file_path)
        rows.append(
            {
                "file_path": str(rel),
                "category": rel.parts[0] if rel.parts else "",
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "encoding": encoding,
                "sha256": sha,
            }
        )
    return pd.DataFrame(rows)


def _first_present(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"No se encontró ninguna columna entre: {names}")


def normalize_precipitation_monthly(bronze_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted((bronze_dir / "precipitacion").glob("*mensual*.csv")):
        df = read_csv_flexible(path)
        df.columns = normalize_columns(df.columns)
        date_col = _first_present(df, ["fecha"])
        value_col = _first_present(df, ["valor"])
        tmp = pd.DataFrame(
            {
                "month": pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                "precip_mm": pd.to_numeric(df[value_col], errors="coerce"),
                "source_file": path.name,
            }
        )
        year_match = re.search(r"(\d{4})", path.name)
        tmp["source_start_year"] = int(year_match.group(1)) if year_match else 0
        frames.append(tmp)
    if not frames:
        raise FileNotFoundError("No se encontraron archivos de precipitación mensual en bronze")
    merged = pd.concat(frames, ignore_index=True).dropna(subset=["month"]).sort_values(["month", "source_start_year"], ascending=[True, False])
    merged = merged.drop_duplicates(subset=["month"], keep="first").sort_values("month").reset_index(drop=True)
    return merged[["month", "precip_mm", "source_file"]]


def normalize_temperature_monthly(bronze_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    temp_dir = bronze_dir / "temperatura"
    min_df = read_csv_flexible(next(temp_dir.glob("*minima*.csv")))
    max_df = read_csv_flexible(next(temp_dir.glob("*maxima*.csv")))
    min_df.columns = normalize_columns(min_df.columns)
    max_df.columns = normalize_columns(max_df.columns)

    def to_monthly(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "month": pd.to_datetime(df["fecha"], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                value_name: pd.to_numeric(df["valor"], errors="coerce"),
            }
        )
        return out.groupby("month", as_index=False).agg({value_name: "mean",})

    tmin = to_monthly(min_df, "tmin_c")
    tmax = to_monthly(max_df, "tmax_c")
    monthly = tmin.merge(tmax, on="month", how="outer").sort_values("month").reset_index(drop=True)
    monthly["tmean_c"] = monthly[["tmin_c", "tmax_c"]].mean(axis=1)
    return monthly


def normalize_oni_monthly(bronze_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    oni_path = next((bronze_dir / "enso").glob("*oni*.csv"))
    oni = read_csv_flexible(oni_path)
    oni.columns = normalize_columns(oni.columns)
    month_col = _first_present(oni, ["mes_central"])
    oni_col = _first_present(oni, ["anomalia_oni", "oni", "oni_anomalia"])
    out = pd.DataFrame(
        {
            "month": pd.to_datetime(oni[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
            "oni_anomaly": pd.to_numeric(oni[oni_col], errors="coerce"),
            "season": oni.get("temporada"),
            "year": pd.to_numeric(oni.get("anio"), errors="coerce"),
        }
    )
    return out.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)


def _normalize_eva(path: Path, schema_version: str) -> pd.DataFrame:
    df = read_csv_flexible(path)
    df.columns = normalize_columns(df.columns)

    col_map = {
        "dane_depto": ["codigo_dane_departamento", "cod_dep"],
        "departamento": ["departamento"],
        "dane_municipio": ["codigo_dane_municipio", "cod_mun"],
        "municipio": ["municipio"],
        "grupo_cultivo": ["grupo_cultivo", "grupo_de_cultivo"],
        "subgrupo_cultivo": ["subgrupo", "subgrupo_de_cultivo"],
        "cultivo": ["cultivo"],
        "desagregacion_cultivo": ["desagregacion_cultivo", "desagregacion_regional_y_o_sistema_productivo"],
        "year": ["ano", "anoo", "a_o", "a_o_", "a_o__", "ano_", "a_o___", "a_o____", "ano__", "a_o_____", "a_o______", "ano___", "a_o_______", "ano____", "a_o________", "ano_____", "a_o_________", "ano______", "a_o__________", "ano_______", "a_o___________", "a_o____________", "a_o_____________", "a_o______________", "a_o_______________", "a_o________________", "a_o_________________", "a_o__________________", "a_o___________________", "a_o____________________", "a_o_____________________", "a_o______________________", "a_o_______________________", "a_o________________________", "a_o_________________________", "a_o__________________________", "a_o___________________________", "a_o____________________________", "a_o_____________________________", "a_o______________________________", "a_o_______________________________", "a_o________________________________", "a_o_________________________________", "a_o__________________________________", "a_o___________________________________", "a_o____________________________________", "a_o_____________________________________", "a_o______________________________________", "a_o_______________________________________", "a_o________________________________________", "a_o_________________________________________", "a_o__________________________________________", "a_o___________________________________________", "a_o____________________________________________", "a_o_____________________________________________", "a_o______________________________________________", "a_o_______________________________________________", "a_o________________________________________________", "a_o_________________________________________________", "a_o__________________________________________________", "a_o___________________________________________________", "a_o____________________________________________________", "a_o_____________________________________________________", "a_o______________________________________________________", "a_o_______________________________________________________", "a_o________________________________________________________", "a_o_________________________________________________________", "a_o__________________________________________________________", "a_o___________________________________________________________", "a_o____________________________________________________________", "a_o_____________________________________________________________", "a_o______________________________________________________________", "a_o_______________________________________________________________", "a_o________________________________________________________________", "a_o_________________________________________________________________", "a_o__________________________________________________________________", "a_o___________________________________________________________________", "a_o____________________________________________________________________", "a_o_____________________________________________________________________", "a_o______________________________________________________________________", "a_o_______________________________________________________________________", "a_o________________________________________________________________________", "a_o_________________________________________________________________________", "a_o__________________________________________________________________________", "a_o___________________________________________________________________________", "a_o____________________________________________________________________________", "a_o_____________________________________________________________________________", "a_o______________________________________________________________________________", "a_o_______________________________________________________________________________", "a_o________________________________________________________________________________", "a_o_________________________________________________________________________________", "a_o__________________________________________________________________________________", "a_o___________________________________________________________________________________", "a_o____________________________________________________________________________________", "a_o_____________________________________________________________________________________", "a_o______________________________________________________________________________________", "a_o_______________________________________________________________________________________", "a_o________________________________________________________________________________________", "a_o_________________________________________________________________________________________", "a_o__________________________________________________________________________________________", "a_o___________________________________________________________________________________________", "a_o____________________________________________________________________________________________", "a_o_____________________________________________________________________________________________", "a_o______________________________________________________________________________________________", "a_o_______________________________________________________________________________________________", "ano", "ano", "a_o", "a_o", "ano"],
        "periodo": ["periodo"],
        "area_sembrada_ha": ["area_sembrada", "area_sembrada_ha"],
        "area_cosechada_ha": ["area_cosechada", "area_cosechada_ha"],
        "produccion_t": ["produccion", "produccion_t"],
        "rendimiento_t_ha": ["rendimiento", "rendimiento_t_ha"],
        "ciclo_cultivo": ["ciclo_del_cultivo", "ciclo_de_cultivo"],
        "estado_fisico": ["estado_fisico_del_cultivo", "estado_fisico_produccion"],
        "nombre_cientifico": ["nombre_cientifico_del_cultivo", "nombre_cientifico"],
    }

    # Limpia llaves redundantes para año y localiza la columna real
    year_col = None
    for candidate in ["ano", "a_o", "a_o_", "a_o__", "a_o___"]:
        if candidate in df.columns:
            year_col = candidate
            break
    if year_col is None:
        for c in df.columns:
            if c.startswith("a") and "o" in c and len(c) <= 6:
                year_col = c
                break
    if year_col:
        col_map["year"] = [year_col]

    out = pd.DataFrame()
    for target, options in col_map.items():
        source = None
        for opt in options:
            if opt in df.columns:
                source = opt
                break
        out[target] = df[source] if source else np.nan

    out["schema_version"] = schema_version
    out["source_file"] = path.name

    out["dane_depto"] = out["dane_depto"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(2)
    out["dane_municipio"] = out["dane_municipio"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    out["municipio"] = out["municipio"].astype(str).str.upper().str.strip()
    out["departamento"] = out["departamento"].astype(str).str.upper().str.strip()
    out["periodo"] = out["periodo"].astype(str).str.upper().str.strip()
    out["year"] = out["year"].map(parse_year).astype("Int64")

    for col in ["area_sembrada_ha", "area_cosechada_ha", "produccion_t", "rendimiento_t_ha"]:
        out[col] = out[col].map(parse_localized_number)

    out["year_start"] = pd.to_datetime(out["year"].astype("string") + "-01-01", errors="coerce")
    return out


def normalize_eva_agriculture(bronze_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    agri_dir = bronze_dir / "agricultura"
    frames = []
    for path in sorted(agri_dir.glob("*.csv")):
        version = "2006_2018" if "2006-2018" in path.name else "2019_2025"
        frames.append(_normalize_eva(path, version))
    if not frames:
        raise FileNotFoundError("No se encontraron archivos EVA en bronze/agricultura")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["year", "dane_municipio", "cultivo"], na_position="last").reset_index(drop=True)


def create_silver_dictionary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in outputs.items():
        for col, dtype in df.dtypes.items():
            rows.append(
                {
                    "dataset": name,
                    "column": col,
                    "dtype": str(dtype),
                    "null_count": int(df[col].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def run_bronze_to_silver(bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR) -> dict[str, pd.DataFrame]:
    ensure_directories()
    silver_dir.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_bronze_manifest(bronze_dir)
    precip = normalize_precipitation_monthly(bronze_dir)
    temp = normalize_temperature_monthly(bronze_dir)
    oni = normalize_oni_monthly(bronze_dir)
    eva = normalize_eva_agriculture(bronze_dir)

    outputs = {
        "manifest_sources": manifest,
        "precipitacion_chicoral_mensual": precip,
        "temperatura_chicoral_mensual": temp,
        "oni_mensual": oni,
        "agricultura_eva_normalizada": eva,
    }

    manifest.to_csv(silver_dir / "manifest_sources.csv", index=False)
    precip.to_csv(silver_dir / "precipitacion_chicoral_mensual.csv", index=False)
    temp.to_csv(silver_dir / "temperatura_chicoral_mensual.csv", index=False)
    oni.to_csv(silver_dir / "oni_mensual.csv", index=False)
    eva.to_csv(silver_dir / "agricultura_eva_normalizada.csv", index=False)

    dictionary = create_silver_dictionary(outputs)
    dictionary.to_csv(silver_dir / "data_dictionary_silver.csv", index=False)
    return outputs

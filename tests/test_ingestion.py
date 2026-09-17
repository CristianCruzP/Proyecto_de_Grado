from pathlib import Path

import pandas as pd

from src.data_ingestion import run_bronze_to_silver


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_bronze_to_silver_normalizes_and_deduplicates(tmp_path):
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    _write(
        bronze / "precipitacion" / "precipitacion_chicoral_mensual_1965-2000.csv",
        "Fecha,Valor
2001-01-01,10
2001-02-01,20
",
    )
    _write(
        bronze / "precipitacion" / "precipitacion_chicoral_mensual_2001-2026.csv",
        "Fecha,Valor
2001-01-01,11
2001-03-01,30
",
    )
    _write(
        bronze / "temperatura" / "temperatura_minima_2000-2026.csv",
        "Fecha,Valor
2001-01-01,20
2001-01-02,22
",
    )
    _write(
        bronze / "temperatura" / "temperatura_maxima_2000-2026.csv",
        "Fecha,Valor
2001-01-01,30
2001-01-02,32
",
    )
    _write(
        bronze / "enso" / "oni_1966-2026.csv",
        "temporada,anio,sst,anomalia_oni,mes_central
DJF,2001,0,1.2,2001-01
",
    )
    _write(
        bronze / "agricultura" / "evaluaciones_agropecuarias_municipales_EVA_2006-2018.csv",
        '"CÓD. MUN.","MUNICIPIO","AÑO","PERIODO","Área Sembrada\n(ha)","Área Cosechada\n(ha)","Producción\n(t)","Rendimiento\n(t/ha)"
"73.268","ESPINAL","2.007","2007A","1.000","900","2.000","2,0"
',
    )
    _write(
        bronze / "agricultura" / "evaluaciones_agropecuarias_municipales_EVA_2019-2025.csv",
        "Código Dane municipio,Municipio,Año,Periodo,Área sembrada,Área cosechada,Producción,Rendimiento
73268,Espinal,2019,2019A,90,80,100,1,25
".replace(",1,25",',"1,25"'),
    )

    outputs = run_bronze_to_silver(bronze_dir=bronze, silver_dir=silver)

    precip = outputs["precipitacion_chicoral_mensual"]
    assert precip["month"].nunique() == len(precip)
    assert float(precip.loc[precip["month"] == pd.Timestamp("2001-01-01"), "precip_mm"].iloc[0]) == 11.0

    eva = outputs["agricultura_eva_normalizada"]
    assert "year" in eva.columns
    assert int(eva["year"].dropna().iloc[0]) >= 2007
    assert (silver / "manifest_sources.csv").exists()

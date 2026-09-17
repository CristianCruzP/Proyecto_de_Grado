# Proyecto de Grado - Sequías LSTM (CRISP-DM)

Implementación reproducible Bronce → Plata → Oro para clima y sequía en Chicoral/Espinal.

## Estructura

- `src/`: lógica reusable (ingestión, calidad, ingeniería, preprocesamiento LSTM, modelado).
- `notebooks/`: narrativa/orquestación CRISP-DM (01 a 05).
- `data/bronze`: fuentes crudas (no se modifican).
- `data/silver`: datos homologados + control de calidad.
- `data/gold`: datasets finales para modelado/evaluación.

## Ejecución rápida

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Para preparar datos con notebooks, ejecutar en orden:

1. `notebooks/01_business_understanding.ipynb`
2. `notebooks/02_data_understanding.ipynb`
3. `notebooks/03_data_preparation.ipynb`
4. `notebooks/04_modeling.ipynb`
5. `notebooks/05_evaluation.ipynb`

## Decisiones clave

- No se rellenan faltantes de forma silenciosa: se conservan `NaN` y se reportan.
- La integración de precipitación mensual evita duplicar meses.
- Escalado con `StandardScaler` ajustado solo con train.
- Ventanas LSTM construidas cronológicamente para evitar fuga temporal.
- EVA 2006–2018 y 2019–2025 se normalizan en una tabla Silver común; en Gold se deja auxiliar anual.

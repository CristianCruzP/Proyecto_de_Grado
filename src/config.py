from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_QUALITY_DIR = REPORTS_DIR / "data_quality"
CRISP_DM_DIR = REPORTS_DIR / "crisp_dm"
MODELS_DIR = ROOT_DIR / "models"
PREPROCESSORS_DIR = MODELS_DIR / "preprocessors"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"


def ensure_directories() -> None:
    for path in [
        SILVER_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        DATA_QUALITY_DIR,
        CRISP_DM_DIR,
        MODELS_DIR,
        PREPROCESSORS_DIR,
        TRAINED_MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

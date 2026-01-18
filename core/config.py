# core/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ==============================================================================
# Rutas del proyecto
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../dashboard_entradas_streamlit
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
CURATED_DIR = DATA_DIR / "curated"


# ==============================================================================
# Archivo fuente (por defecto)
# ==============================================================================

DEFAULT_EXCEL_FILENAME = "Base de datos Entradas Almacen 0720.xlsx"


def default_input_excel_path() -> Path:
    return INPUT_DIR / DEFAULT_EXCEL_FILENAME


def curated_parquet_path() -> Path:
    return CURATED_DIR / "entradas_curated.parquet"


def curated_bad_parquet_path() -> Path:
    return CURATED_DIR / "entradas_rechazadas.parquet"


# ==============================================================================
# Defaults de dashboard
# ==============================================================================

@dataclass(frozen=True)
class AppDefaults:
    # años que más se usan (puedes ampliar)
    years_focus: tuple[int, ...] = (2024, 2025)

    # top n por defecto
    top_skus: int = 15

    # filtros iniciales
    default_doc_prefixes: tuple[str, ...] = ("STOCK", "VT")
    include_otros_by_default: bool = False  # normalmente el enfoque es PRIMERA/SEGUNDA

    # umbrales para alertas simples (se usarán luego)
    pct_segunda_alert: float = 20.0   # % segunda mensual o diaria
    piezas_dia_alert: float = 0.0     # si quieres activar por percentil, lo haremos en Alertas


DEFAULTS = AppDefaults()


# ==============================================================================
# Utilidades
# ==============================================================================

def ensure_dirs() -> None:
    """Crea carpetas base si no existen."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

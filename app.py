# app.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from core.config import (
    ensure_dirs,
    INPUT_DIR,
    CURATED_DIR,
    DEFAULT_EXCEL_FILENAME,
    curated_parquet_path,
    curated_bad_parquet_path,
)
from core.io_excel import load_entradas_excel
from core.transform import build_aggs


# ==============================================================================
# Caching
# ==============================================================================

@st.cache_data(show_spinner=False)
def _load_excel_cached(path_str: str, mtime: float):
    df_ok, df_bad = load_entradas_excel(Path(path_str))
    return df_ok, df_bad


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path_str: str, mtime: float):
    return pd.read_parquet(path_str)


def _try_save_parquet(df: pd.DataFrame, out_path: Path) -> tuple[bool, str]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return True, ""
    except Exception as e:
        return False, str(e)


def _try_read_parquet(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


# ==============================================================================
# UI
# ==============================================================================

st.set_page_config(page_title="Dashboard Entradas Almacén", page_icon="📦", layout="wide")

ensure_dirs()

st.title("📦 Dashboard Entradas Almacén")
st.caption("Reglas: SEGUNDA = SANSON + POP | PRIMERA = resto (incluye OTROS como PRIMERA)")


# ==============================================================================
# Detectar archivos disponibles
# ==============================================================================
xlsx_files = sorted([p for p in INPUT_DIR.glob("*.xlsx") if p.is_file()])

st.sidebar.header("Fuente de datos")

if xlsx_files:
    default_idx = 0
    for i, p in enumerate(xlsx_files):
        if p.name == DEFAULT_EXCEL_FILENAME:
            default_idx = i
            break

    selected_file = st.sidebar.selectbox(
        "Archivo Excel en data/input",
        options=xlsx_files,
        index=default_idx,
        format_func=lambda p: p.name,
    )
else:
    selected_file = None
    st.sidebar.info("No hay .xlsx en data/input. Puedes subir uno aquí abajo.")

use_curated = st.sidebar.toggle(
    "Usar curado (parquet) si existe",
    value=True,
    help="Si existe data/curated/entradas_curated.parquet, carga más rápido.",
)

force_rebuild = st.sidebar.toggle(
    "Forzar recarga desde Excel",
    value=False,
    help="Ignora el parquet y reconstruye desde el Excel seleccionado.",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 input:  {INPUT_DIR}")
st.sidebar.caption(f"📁 curated: {CURATED_DIR}")


# ==============================================================================
# Upload opcional (si no hay archivos)
# ==============================================================================
uploaded = None
if selected_file is None:
    uploaded = st.file_uploader("Sube el Excel histórico de entradas (.xlsx)", type=["xlsx"])


# ==============================================================================
# Carga (parquet o excel)
# ==============================================================================
df_ok = None
df_bad = None
source_info = {}

with st.spinner("Cargando datos..."):
    if uploaded is not None:
        # Guardar temporalmente el upload en data/input para tener pipeline estable
        tmp_path = INPUT_DIR / uploaded.name
        tmp_path.write_bytes(uploaded.getbuffer())
        selected_file = tmp_path

    if selected_file is None:
        st.warning("Coloca el archivo Excel en data/input o súbelo arriba para continuar.")
        st.stop()

    curated_ok = curated_parquet_path()
    curated_bad = curated_bad_parquet_path()

    # 1) Intentar parquet
    if use_curated and curated_ok.exists() and (not force_rebuild):
        df_ok_try = _try_read_parquet(curated_ok)
        if df_ok_try is not None:
            df_ok = df_ok_try
            df_bad = _try_read_parquet(curated_bad) if curated_bad.exists() else pd.DataFrame()
            source_info = {"source": "parquet", "path": str(curated_ok)}
        else:
            # si falla lectura, reconstruir
            force_rebuild = True

    # 2) Reconstruir desde Excel
    if df_ok is None:
        mtime = selected_file.stat().st_mtime
        df_ok, df_bad = _load_excel_cached(str(selected_file), mtime)

        # Guardar parquet (si se puede)
        ok_saved, ok_err = _try_save_parquet(df_ok, curated_ok)
        bad_saved, bad_err = _try_save_parquet(df_bad, curated_bad) if df_bad is not None else (True, "")

        source_info = {
            "source": "excel",
            "path": str(selected_file),
            "parquet_saved": ok_saved and bad_saved,
            "parquet_error": (ok_err or bad_err) if not (ok_saved and bad_saved) else "",
        }

# Agregados (esto también asegura que existan: producto_norm, clasificacion_producto, calidad)
aggs = build_aggs(df_ok)

# Guardar en session_state para multipage
st.session_state["df_ok"] = df_ok
st.session_state["df_bad"] = df_bad if df_bad is not None else pd.DataFrame()
st.session_state["aggs"] = aggs
st.session_state["source_info"] = source_info


# ==============================================================================
# Home KPIs
# ==============================================================================
base = aggs["base"].copy()

min_d = pd.to_datetime(base["dia"]).min()
max_d = pd.to_datetime(base["dia"]).max()

total_piezas = float(pd.to_numeric(base["piezas"], errors="coerce").fillna(0).sum())
primera = float(base.loc[base["calidad"] == "PRIMERA", "piezas"].sum())
segunda = float(base.loc[base["calidad"] == "SEGUNDA", "piezas"].sum())
total_ps = primera + segunda
pct_seg = (segunda / total_ps * 100) if total_ps > 0 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Registros válidos", f"{len(base):,}")
c2.metric("Piezas total", f"{int(round(total_piezas)):,}")
c3.metric("PRIMERA", f"{int(round(primera)):,}")
c4.metric("SEGUNDA", f"{int(round(segunda)):,}")
c5.metric("% SEGUNDA", f"{pct_seg:.2f}%")
c6.metric("SKUs distintos", f"{base['sku'].nunique():,}")

st.markdown("---")

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Estado de la carga")
    st.write(
        {
            "origen": source_info.get("source"),
            "archivo": source_info.get("path"),
            "rango_fechas": f"{min_d.date()} → {max_d.date()}",
        }
    )
    if source_info.get("source") == "excel":
        if source_info.get("parquet_saved"):
            st.success("Curado guardado en data/curated (parquet).")
        else:
            st.warning("No pude guardar parquet (no crítico).")
            if source_info.get("parquet_error"):
                st.caption(f"Detalle: {source_info['parquet_error']}")

    # snapshot de clasificación (útil para dirección)
    if "clasificacion_producto" in base.columns:
        st.subheader("Mix por clasificación (Top 12 por volumen)")
        mix = (
            base.groupby("clasificacion_producto", as_index=False)["piezas"]
            .sum()
            .sort_values("piezas", ascending=False)
            .head(12)
        )
        st.dataframe(mix, use_container_width=True)

    st.subheader("Preview (base)")
    st.dataframe(base.head(25), use_container_width=True)

with right:
    st.subheader("Rechazados (diagnóstico rápido)")
    bad = st.session_state["df_bad"]
    if bad is None or bad.empty:
        st.success("No hay filas rechazadas 🎉")
    else:
        # top motivos
        if "motivo" in bad.columns:
            top = bad["motivo"].fillna("NA").value_counts().head(12).reset_index()
            top.columns = ["motivo", "filas"]
            st.dataframe(top, use_container_width=True)

        st.caption("Muestra de rechazados")
        st.dataframe(bad.head(20), use_container_width=True)

st.info("➡️ Usa el menú de la izquierda (páginas) para ver: Resumen Ejecutivo, Producción Diaria, Calidad, SKU Detail y Alertas.")

# pages/01_Resumen_Ejecutivo.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.charts import (
    chart_anual_pct,
    chart_mensual_stacked,
    chart_pct_segunda_mensual,
)
from core.transform import build_aggs

try:
    from core.rules import CLASIFICACION_ORDER
except Exception:
    CLASIFICACION_ORDER = ["SANSON", "POP", "GALV_BISA", "PH", "CF", "TG", "IMU", "OTROS", "BISA"]


st.set_page_config(page_title="Resumen Ejecutivo", page_icon="📊", layout="wide")
st.title("📊 Resumen Ejecutivo")


# ==============================================================================
# Carga desde session_state
# ==============================================================================
df_ok = st.session_state.get("df_ok")
aggs = st.session_state.get("aggs")

if df_ok is None:
    st.warning("No encontré df_ok en session_state. Vuelve a cargar datos desde app.py.")
    st.stop()

if aggs is None:
    aggs = build_aggs(df_ok)
    st.session_state["aggs"] = aggs

base = aggs["base"].copy()
base["dia"] = pd.to_datetime(base["dia"], errors="coerce")

min_d = base["dia"].min()
max_d = base["dia"].max()


# ==============================================================================
# Filtros
# ==============================================================================
st.sidebar.header("Filtros")

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)

start = pd.Timestamp(date_range[0])
end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # inclusive

use_clasif = "clasificacion_producto" in base.columns
if not use_clasif:
    st.sidebar.warning(
        "No encuentro la columna 'clasificacion_producto'. "
        "Asegúrate de haber actualizado core/rules.py y recargado datos. "
        "Usaré doc_prefix como fallback."
    )

if use_clasif:
    present = (
        base["clasificacion_producto"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )

    ordered = [c for c in CLASIFICACION_ORDER if c in present]
    extras = sorted([c for c in present if c not in set(CLASIFICACION_ORDER)])
    options = ordered + extras if (ordered or extras) else CLASIFICACION_ORDER

    sel_clasif = st.sidebar.multiselect(
        "Clasificación (por NOMBRE PRODUCTO)",
        options=options,
        default=options,
        help="SANSON y POP son SEGUNDA. El resto cuenta como PRIMERA (incluye OTROS).",
    )
else:
    doc_prefixes = sorted(base["doc_prefix"].dropna().astype(str).unique().tolist()) if "doc_prefix" in base.columns else []
    sel_doc = st.sidebar.multiselect(
        "doc_prefix (fallback)",
        options=doc_prefixes,
        default=doc_prefixes,
    )

mask = (base["dia"] >= start) & (base["dia"] < end)

if use_clasif:
    if sel_clasif:
        mask &= base["clasificacion_producto"].astype(str).isin(sel_clasif)
else:
    if "doc_prefix" in base.columns and sel_doc:
        mask &= base["doc_prefix"].astype(str).isin(sel_doc)

base_f = base[mask].copy()

# Recalcular aggs sobre base filtrada
aggs_f = build_aggs(base_f)
anual = aggs_f["anual"]
mensual = aggs_f["mensual"]

years = sorted(anual["anio"].dropna().astype(int).unique().tolist()) if not anual.empty else []
sel_year = st.sidebar.selectbox(
    "Año (para vistas mensuales)",
    options=years,
    index=(len(years) - 1) if years else 0,
    disabled=(len(years) == 0),
)


# ==============================================================================
# KPIs
# ==============================================================================
st.subheader("KPIs del período filtrado")

total_piezas = float(pd.to_numeric(base_f["piezas"], errors="coerce").fillna(0).sum())
primera = float(base_f.loc[base_f["calidad"] == "PRIMERA", "piezas"].sum()) if "calidad" in base_f.columns else 0.0
segunda = float(base_f.loc[base_f["calidad"] == "SEGUNDA", "piezas"].sum()) if "calidad" in base_f.columns else 0.0
total_ps = primera + segunda
pct_seg = (segunda / total_ps * 100) if total_ps > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Piezas (total)", f"{int(round(total_piezas)):,}")
c2.metric("Piezas PRIMERA", f"{int(round(primera)):,}")
c3.metric("Piezas SEGUNDA", f"{int(round(segunda)):,}")
c4.metric("% SEGUNDA", f"{pct_seg:.2f}%")
c5.metric("SKUs distintos", f"{base_f['sku'].nunique():,}")


# ==============================================================================
# Gráficas
# ==============================================================================
st.markdown("---")
st.subheader("Mix de calidad (anual en %)")

if anual.empty:
    st.warning("No hay datos suficientes para calcular anual.")
else:
    fig = chart_anual_pct(anual)
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.subheader(f"Producción mensual {sel_year}: PRIMERA vs SEGUNDA")

if sel_year is None or mensual.empty:
    st.warning("No hay datos mensuales para mostrar.")
else:
    fig_m = chart_mensual_stacked(mensual, int(sel_year))
    st.pyplot(fig_m, use_container_width=True)

st.subheader(f"% SEGUNDA mensual {sel_year}")

if sel_year is None or mensual.empty or "pct_segunda" not in mensual.columns:
    st.warning("No hay datos suficientes para % segunda mensual.")
else:
    fig_p = chart_pct_segunda_mensual(mensual, int(sel_year))
    st.pyplot(fig_p, use_container_width=True)


# ==============================================================================
# Tablas para directivos
# ==============================================================================
st.markdown("---")
st.subheader("Tabla anual (para copiar a reporte)")

if not anual.empty:
    cols = ["anio", "piezas_primera", "piezas_segunda", "piezas_total_ps", "pct_segunda"]
    cols = [c for c in cols if c in anual.columns]
    t = anual[cols].copy().sort_values("anio")
    st.dataframe(t, use_container_width=True)

st.subheader("Tabla mensual (últimos 18 meses)")

if not mensual.empty:
    cols2 = ["anio", "mes", "piezas_primera", "piezas_segunda", "piezas_total_ps", "pct_segunda"]
    cols2 = [c for c in cols2 if c in mensual.columns]
    t2 = mensual[cols2].copy().sort_values(["anio", "mes"])
    t2 = t2.tail(18).reset_index(drop=True)
    st.dataframe(t2, use_container_width=True)

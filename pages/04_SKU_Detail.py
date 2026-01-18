# pages/04_SKU_Detail.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.transform import build_aggs

try:
    from core.rules import CLASIFICACION_ORDER
except Exception:
    CLASIFICACION_ORDER = ["SANSON", "POP", "GALV_BISA", "PH", "CF", "TG", "IMU", "OTROS", "BISA"]


st.set_page_config(page_title="SKU Detail", page_icon="🔎", layout="wide")
st.title("🔎 SKU Detail")


# ==============================================================================
# Carga desde session_state
# ==============================================================================
df_ok = st.session_state.get("df_ok")
aggs = st.session_state.get("aggs")

if df_ok is None:
    st.error("No encontré df_ok en session_state. Abre primero la Home (app.py).")
    st.stop()

if aggs is None:
    aggs = build_aggs(df_ok)
    st.session_state["aggs"] = aggs

base = aggs["base"].copy()
base["dia"] = pd.to_datetime(base["dia"], errors="coerce")
base["fecha_dt"] = pd.to_datetime(base.get("fecha_dt", pd.NaT), errors="coerce")
base["piezas"] = pd.to_numeric(base["piezas"], errors="coerce").fillna(0)
base["producto"] = base.get("producto", "").fillna("").astype(str)

base = base[(base["piezas"] > 0) & base["dia"].notna()].copy()

use_clasif = "clasificacion_producto" in base.columns
if use_clasif:
    base["clasificacion_producto"] = base["clasificacion_producto"].fillna("").astype(str).str.strip()
else:
    base["doc_prefix"] = base.get("doc_prefix", "NA").fillna("NA").astype(str).str.strip()


# ==============================================================================
# Sidebar filtros
# ==============================================================================
st.sidebar.header("Filtros")

min_d = base["dia"].min()
max_d = base["dia"].max()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)

start = pd.Timestamp(date_range[0])
end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

# ---- filtro principal: clasificacion_producto (sustituye doc_prefix) ----
if use_clasif:
    present = (
        base["clasificacion_producto"]
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
    doc_prefixes = sorted(base["doc_prefix"].dropna().astype(str).unique().tolist())
    sel_doc = st.sidebar.multiselect("doc_prefix (fallback)", options=doc_prefixes, default=doc_prefixes)

# aplicar filtros base
mask = (base["dia"] >= start) & (base["dia"] < end)

if use_clasif:
    if sel_clasif:
        mask &= base["clasificacion_producto"].isin(sel_clasif)
else:
    if sel_doc:
        mask &= base["doc_prefix"].isin(sel_doc)

bf = base[mask].copy()

# En este dashboard, calidad ya es PRIMERA/SEGUNDA (OTROS cuenta como PRIMERA)
bf = bf[bf["calidad"].isin(["PRIMERA", "SEGUNDA"])].copy()

if bf.empty:
    st.warning("Con estos filtros no quedan datos.")
    st.stop()


# ==============================================================================
# Selector de SKU (SKU — Producto corto) + búsqueda
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("SKU")

mode = st.sidebar.radio(
    "Buscar por",
    options=["SKU o Producto", "SKU", "Producto"],
    index=0,
)

q = st.sidebar.text_input(
    "Buscar (escribe y filtra coincidencias)",
    value="",
    help="Filtra la lista del selector. Ej: 'BUDINERA', 'IMU', '2106', 'GALV', etc.",
)

# Ranking: los SKUs más movidos arriba + producto representativo (modo)
def _mode_text(s: pd.Series) -> str:
    s = s.fillna("").astype(str)
    s = s[s.str.strip() != ""]
    if len(s) == 0:
        return ""
    return s.value_counts().index[0]

sku_meta = (
    bf.groupby("sku", as_index=False)
    .agg(
        piezas=("piezas", "sum"),
        producto_display=("producto", _mode_text),
    )
    .sort_values("piezas", ascending=False)
    .reset_index(drop=True)
)

sku_meta["sku"] = sku_meta["sku"].astype(str)
sku_meta["producto_norm"] = sku_meta["producto_display"].astype(str).str.upper()

def _short(s: str, n: int = 60) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")

sku_meta["producto_corto"] = sku_meta["producto_display"].apply(_short)

# filtro por búsqueda
if q.strip():
    qn = q.strip().upper()
    if mode == "SKU":
        keep = sku_meta["sku"].str.upper().str.contains(qn, na=False)
    elif mode == "Producto":
        keep = sku_meta["producto_norm"].str.contains(qn, na=False)
    else:
        keep = (
            sku_meta["sku"].str.upper().str.contains(qn, na=False)
            | sku_meta["producto_norm"].str.contains(qn, na=False)
        )
    sku_meta_f = sku_meta[keep].copy()
else:
    sku_meta_f = sku_meta

if sku_meta_f.empty:
    st.sidebar.warning("No hay SKUs que coincidan con tu búsqueda (o con los filtros).")
    st.stop()

label_map = dict(
    zip(
        sku_meta_f["sku"].tolist(),
        (sku_meta_f["sku"] + " — " + sku_meta_f["producto_corto"]).tolist(),
    )
)

sku_list = sku_meta_f["sku"].tolist()

sku_selected = st.sidebar.selectbox(
    "Selecciona SKU",
    options=sku_list,
    index=0,
    format_func=lambda x: label_map.get(str(x), str(x)),
)

df_sku = bf[bf["sku"].astype(str) == str(sku_selected)].copy()

if df_sku.empty:
    st.warning("No hay datos para ese SKU con los filtros actuales.")
    st.stop()


# ==============================================================================
# KPIs del SKU
# ==============================================================================
total = float(df_sku["piezas"].sum())
primera = float(df_sku.loc[df_sku["calidad"] == "PRIMERA", "piezas"].sum()) if "calidad" in df_sku.columns else 0.0
segunda = float(df_sku.loc[df_sku["calidad"] == "SEGUNDA", "piezas"].sum()) if "calidad" in df_sku.columns else 0.0
total_ps = primera + segunda
pct_seg = (segunda / total_ps * 100) if total_ps > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Piezas (SKU)", f"{int(round(total)):,}")
c2.metric("PRIMERA", f"{int(round(primera)):,}")
c3.metric("SEGUNDA", f"{int(round(segunda)):,}")
c4.metric("% SEGUNDA", f"{pct_seg:.2f}%")
c5.metric("Días activos", f"{df_sku['dia'].nunique():,}")

st.caption("Reglas: SEGUNDA = SANSON + POP | PRIMERA = resto (incluye OTROS)")


# ==============================================================================
# 1) Tendencia mensual del SKU (stacked)
# ==============================================================================
st.markdown("---")
st.subheader("Tendencia mensual (PRIMERA vs SEGUNDA)")

df_sku["mes"] = df_sku["dia"].dt.to_period("M").dt.to_timestamp()

g = df_sku.groupby(["mes", "calidad"], as_index=False)["piezas"].sum()
piv = (
    g.pivot_table(index="mes", columns="calidad", values="piezas", aggfunc="sum", fill_value=0)
    .reset_index()
)

if "PRIMERA" not in piv.columns:
    piv["PRIMERA"] = 0.0
if "SEGUNDA" not in piv.columns:
    piv["SEGUNDA"] = 0.0

piv = piv.sort_values("mes")

fig1, ax1 = plt.subplots(figsize=(11.5, 5.2), dpi=220)
ax1.bar(piv["mes"], piv["PRIMERA"], label="PRIMERA")
ax1.bar(piv["mes"], piv["SEGUNDA"], bottom=piv["PRIMERA"], label="SEGUNDA")
ax1.set_ylabel("Piezas")
ax1.set_title(f"SKU {sku_selected} - Producción mensual")
ax1.legend()
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", alpha=0.25)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
fig1.tight_layout()
st.pyplot(fig1, use_container_width=True)


# ==============================================================================
# 2) Producción diaria del SKU (línea)
# ==============================================================================
st.subheader("Producción diaria (PRIMERA+SEGUNDA)")

d = (
    df_sku.groupby("dia", as_index=False)["piezas"].sum()
    .sort_values("dia")
)

fig2, ax2 = plt.subplots(figsize=(11.5, 4.8), dpi=220)
ax2.plot(d["dia"], d["piezas"])
ax2.set_ylabel("Piezas")
ax2.set_title(f"SKU {sku_selected} - Producción diaria")
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.grid(axis="y", alpha=0.25)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)


# ==============================================================================
# 3) Corte por Clasificación (sustituye doc_prefix)
# ==============================================================================
st.markdown("---")

group_col = "clasificacion_producto" if use_clasif else "doc_prefix"
title_col = "Clasificación" if use_clasif else "doc_prefix"

st.subheader(f"Corte por {title_col}")

tmp = df_sku.copy()
tmp["is_seg"] = (tmp["calidad"] == "SEGUNDA").astype(int)
tmp["p_seg"] = tmp["piezas"] * tmp["is_seg"]

pref = (
    tmp.groupby(group_col, as_index=False)
    .agg(
        piezas_total=("piezas", "sum"),
        piezas_segunda=("p_seg", "sum"),
        dias=("dia", "nunique"),
        registros=("sku", "size"),
    )
    .sort_values("piezas_total", ascending=False)
)
pref["pct_segunda"] = np.where(
    pref["piezas_total"] > 0,
    pref["piezas_segunda"] / pref["piezas_total"] * 100,
    0.0,
)

st.dataframe(pref, use_container_width=True)

# gráfico rápido
topn = min(10, len(pref))
p = pref.head(topn).iloc[::-1].copy()
fig3, ax3 = plt.subplots(figsize=(10.5, 5.6), dpi=220)
ax3.barh(p[group_col].astype(str), p["piezas_total"])
ax3.set_xlabel("Piezas")
ax3.set_title(f"SKU {sku_selected} - Top {topn} {title_col} por volumen")
ax3.grid(axis="x", alpha=0.25)
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)


# ==============================================================================
# 4) Detalle (tabla)
# ==============================================================================
st.markdown("---")
st.subheader("Detalle de registros (filtrado)")

cols_show = [
    "dia", "fecha_dt", "hora_ok",
    "sku", "piezas", "cajas",
    "calidad",
    "clasificacion_producto",
    "doc_prefix",
    "documento_ref", "producto",
]
cols_show = [c for c in cols_show if c in df_sku.columns]

df_show = df_sku[cols_show].sort_values(["dia", "fecha_dt"], ascending=[False, False])
st.dataframe(df_show, use_container_width=True, height=420)


# ==============================================================================
# Descargas
# ==============================================================================
st.subheader("Descargar reporte del SKU")

csv_detail = df_show.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV (detalle SKU)",
    data=csv_detail,
    file_name=f"sku_{sku_selected}_detalle.csv",
    mime="text/csv",
)

csv_month = piv.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV (mensual SKU)",
    data=csv_month,
    file_name=f"sku_{sku_selected}_mensual.csv",
    mime="text/csv",
)

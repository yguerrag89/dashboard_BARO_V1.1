# pages/03_Calidad.py
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


st.set_page_config(page_title="Calidad", page_icon="✅", layout="wide")
st.title("✅ Calidad (PRIMERA vs SEGUNDA)")


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
base["piezas"] = pd.to_numeric(base["piezas"], errors="coerce").fillna(0)
base["producto"] = base.get("producto", "").fillna("").astype(str)

# Asegura limpieza mínima
base = base[(base["piezas"] > 0) & (base["dia"].notna())].copy()

use_clasif = "clasificacion_producto" in base.columns
if use_clasif:
    base["clasificacion_producto"] = base["clasificacion_producto"].fillna("").astype(str).str.strip()
else:
    # fallback temporal
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

# ---- filtro por clasificación (sustituye doc_prefix) ----
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

# umbrales para top PRODUCTOS
min_piezas_ps = st.sidebar.number_input(
    "Mín piezas (PRIMERA+SEGUNDA) para Top PRODUCTOS",
    min_value=0,
    value=500,
    step=50,
)
min_dias_prod = st.sidebar.number_input(
    "Mín días activos para Top PRODUCTOS",
    min_value=1,
    value=5,
    step=1,
)
top_n = st.sidebar.slider("Top N PRODUCTOS", 5, 50, 15, 1)

alert_pct = st.sidebar.slider("Alerta % segunda mensual (umbral)", 1.0, 50.0, 20.0, 0.5)

# aplicar filtros
mask = (base["dia"] >= start) & (base["dia"] < end)

if use_clasif:
    if sel_clasif:
        mask &= base["clasificacion_producto"].isin(sel_clasif)
else:
    mask &= base["doc_prefix"].isin(sel_doc) if "doc_prefix" in base.columns and sel_doc else True

bf = base[mask].copy()

# foco: calidad (ahora todo es PRIMERA/SEGUNDA por reglas nuevas)
bf = bf[bf["calidad"].isin(["PRIMERA", "SEGUNDA"])].copy()

if bf.empty:
    st.warning("Con estos filtros no quedan datos.")
    st.stop()


# ==============================================================================
# KPIs
# ==============================================================================
primera = float(bf.loc[bf["calidad"] == "PRIMERA", "piezas"].sum())
segunda = float(bf.loc[bf["calidad"] == "SEGUNDA", "piezas"].sum())
total_ps = primera + segunda
pct_seg = (segunda / total_ps * 100) if total_ps > 0 else 0.0

# producto_norm si existe (mejor), si no fallback a producto
if "producto_norm" in bf.columns:
    prod_key = "producto_norm"
else:
    # normalización simple
    bf["producto_norm"] = bf["producto"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    prod_key = "producto_norm"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("PRIMERA (piezas)", f"{int(round(primera)):,}")
c2.metric("SEGUNDA (piezas)", f"{int(round(segunda)):,}")
c3.metric("% SEGUNDA", f"{pct_seg:.2f}%")
c4.metric("Productos analizados", f"{bf[prod_key].nunique():,}")
c5.metric("SKUs distintos", f"{bf['sku'].nunique():,}")


# ==============================================================================
# Agregados sobre filtrado
# ==============================================================================
aggs_f = build_aggs(bf)
mensual = aggs_f["mensual"].copy()
mensual["mes"] = pd.to_datetime(mensual["mes"], errors="coerce")
mensual = mensual.sort_values("mes").reset_index(drop=True)

years = sorted(mensual["anio"].dropna().astype(int).unique().tolist()) if not mensual.empty else []
sel_year = st.selectbox("Año (vista mensual)", options=years, index=len(years) - 1 if years else 0)

m_y = mensual[mensual["anio"] == int(sel_year)].copy()


# ==============================================================================
# 1) % Segunda mensual (línea) + alertas
# ==============================================================================
st.markdown("---")
st.subheader(f"% SEGUNDA mensual {sel_year}")

if m_y.empty:
    st.info("No hay datos mensuales para ese año.")
else:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=220)
    ax.plot(m_y["mes"], m_y["pct_segunda"])
    ax.axhline(float(alert_pct), linestyle="--")
    ax.set_ylabel("% Segunda (sobre PRIMERA+SEGUNDA)")
    ax.set_title(f"% Segunda mensual {sel_year} (umbral alerta: {alert_pct:.1f}%)")
    ax.set_ylim(0, max(5, float(m_y["pct_segunda"].max()) * 1.15, alert_pct * 1.1))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Ranking de meses por % segunda (año seleccionado)")
    rank = m_y[["mes", "piezas_total_ps", "pct_segunda"]].copy()
    rank["mes"] = rank["mes"].dt.strftime("%Y-%m")
    rank = rank.sort_values("pct_segunda", ascending=False).reset_index(drop=True)
    st.dataframe(rank, use_container_width=True)


# ==============================================================================
# 2) Top PRODUCTOS por % segunda (con umbrales)
# ==============================================================================
st.markdown("---")
st.subheader("Top PRODUCTOS por % SEGUNDA (con filtros anti-ruido)")

tmp = bf.copy()
tmp["is_segunda"] = (tmp["calidad"] == "SEGUNDA").astype(int)
tmp["p_seg"] = tmp["piezas"] * tmp["is_segunda"]

# nombre “humano” del producto: el más frecuente dentro del grupo
prod_name = (
    tmp.groupby(prod_key)["producto"]
    .agg(lambda s: s.value_counts().index[0] if len(s) else "")
    .rename("producto_display")
    .reset_index()
)

prod_agg = (
    tmp.groupby(prod_key, as_index=False)
    .agg(
        piezas_segunda=("p_seg", "sum"),
        piezas_total_ps=("piezas", "sum"),
        dias=("dia", "nunique"),
        skus=("sku", "nunique"),
        first_seen=("dia", "min"),
        last_seen=("dia", "max"),
    )
    .merge(prod_name, on=prod_key, how="left")
)

prod_agg["pct_segunda"] = np.where(
    prod_agg["piezas_total_ps"] > 0,
    prod_agg["piezas_segunda"] / prod_agg["piezas_total_ps"] * 100,
    0.0,
)

prod_f = prod_agg[
    (prod_agg["piezas_total_ps"] >= float(min_piezas_ps)) &
    (prod_agg["dias"] >= int(min_dias_prod))
].copy()

prod_f = prod_f.sort_values(["pct_segunda", "piezas_total_ps"], ascending=[False, False]).reset_index(drop=True)

st.caption(f"Filtro aplicado: piezas_total_ps >= {min_piezas_ps} y días >= {min_dias_prod}")

top = prod_f.head(int(top_n)).copy()

# columna corta para tabla
def _short(s: str, n: int = 70) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[: n - 1] + "…")

if not top.empty:
    top["producto_corto"] = top["producto_display"].apply(_short)

st.dataframe(
    top[["producto_corto", "pct_segunda", "piezas_total_ps", "piezas_segunda", "dias", "skus", "first_seen", "last_seen"]]
    .rename(columns={"producto_corto": "producto"}),
    use_container_width=True,
)

# gráfico barras horizontal top
if not top.empty:
    top2 = top.iloc[::-1].copy()
    fig2, ax2 = plt.subplots(figsize=(10.5, 6.2), dpi=220)
    ax2.barh(top2["producto_corto"].astype(str), top2["pct_segunda"])
    ax2.set_xlabel("% Segunda")
    ax2.set_title(f"Top {top_n} PRODUCTOS por % Segunda (filtrado)")
    ax2.grid(axis="x", alpha=0.25)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)


# ==============================================================================
# 3) Comparación por CLASIFICACIÓN (sustituye doc_prefix)
# ==============================================================================
st.markdown("---")

group_col = "clasificacion_producto" if use_clasif else "doc_prefix"
title_col = "Clasificación" if use_clasif else "doc_prefix"

st.subheader(f"Comparación por {title_col} (período filtrado)")

pref = bf.copy()
pref["is_seg"] = (pref["calidad"] == "SEGUNDA").astype(int)
pref["p_seg"] = pref["piezas"] * pref["is_seg"]

pref_agg = (
    pref.groupby(group_col, as_index=False)
    .agg(
        piezas_total=("piezas", "sum"),
        piezas_segunda=("p_seg", "sum"),
        productos=(prod_key, "nunique"),
        skus=("sku", "nunique"),
        dias=("dia", "nunique"),
    )
    .sort_values("piezas_total", ascending=False)
)
pref_agg["pct_segunda"] = np.where(
    pref_agg["piezas_total"] > 0,
    pref_agg["piezas_segunda"] / pref_agg["piezas_total"] * 100,
    0.0,
)

st.dataframe(pref_agg, use_container_width=True)

# gráfico %segunda por clasificación (top 12 por volumen)
top_pref_n = 12
p = pref_agg.head(top_pref_n).iloc[::-1].copy()
fig3, ax3 = plt.subplots(figsize=(10.5, 6.0), dpi=220)
ax3.barh(p[group_col].astype(str), p["pct_segunda"])
ax3.set_xlabel("% Segunda")
ax3.set_title(f"% Segunda por {title_col} (Top {top_pref_n} por volumen)")
ax3.grid(axis="x", alpha=0.25)
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)


# ==============================================================================
# Descargas
# ==============================================================================
st.markdown("---")
st.subheader("Descargas")

csv_top = top.to_csv(index=False).encode("utf-8") if not top.empty else b""
st.download_button(
    "⬇️ Descargar Top PRODUCTOS (%segunda) CSV",
    data=csv_top,
    file_name="top_productos_pct_segunda.csv",
    mime="text/csv",
    disabled=top.empty,
)

csv_pref = pref_agg.to_csv(index=False).encode("utf-8") if not pref_agg.empty else b""
st.download_button(
    f"⬇️ Descargar {title_col} (%segunda) CSV",
    data=csv_pref,
    file_name=f"{title_col.lower()}_pct_segunda.csv".replace(" ", "_"),
    mime="text/csv",
    disabled=pref_agg.empty,
)

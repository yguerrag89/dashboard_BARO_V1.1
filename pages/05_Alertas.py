# pages/05_Alertas.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.transform import build_aggs

try:
    from core.rules import CLASIFICACION_ORDER
except Exception:
    CLASIFICACION_ORDER = ["SANSON", "POP", "GALV_BISA", "PH", "CF", "TG", "IMU", "OTROS", "BISA"]


st.set_page_config(page_title="Alertas", page_icon="🚨", layout="wide")
st.title("🚨 Alertas (Calidad / Volumen)")


# ==============================================================================
# Carga
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

# filtro principal: Clasificación (sustituye doc_prefix)
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
        help="SEGUNDA = SANSON + POP | PRIMERA = resto (incluye OTROS).",
    )
else:
    st.sidebar.warning(
        "No encuentro 'clasificacion_producto'. Usaré doc_prefix como fallback. "
        "Para migrar al 100%, hay que ajustar core/transform.py para recalcular reglas si falta esa columna."
    )
    doc_prefixes = sorted(base["doc_prefix"].dropna().astype(str).unique().tolist())
    sel_doc = st.sidebar.multiselect("doc_prefix (fallback)", options=doc_prefixes, default=doc_prefixes)

# Umbrales
pct_seg_umbral = st.sidebar.slider("%segunda mensual (umbral)", 1.0, 60.0, 20.0, 0.5)

z_thr = st.sidebar.slider("Z-score días atípicos (volumen)", 1.0, 4.0, 2.0, 0.1)
window = st.sidebar.slider("Ventana rolling (días)", 7, 45, 14, 1)

min_piezas_ps = st.sidebar.number_input("Mín piezas PS para alerta SKU", min_value=0, value=500, step=50)
min_dias_sku = st.sidebar.number_input("Mín días para alerta SKU", min_value=1, value=5, step=1)
sku_pct_thr = st.sidebar.slider("%segunda SKU (umbral)", 1.0, 90.0, 30.0, 0.5)
top_n = st.sidebar.slider("Top N SKUs alertados", 5, 100, 30, 1)

# aplicar filtros
mask = (base["dia"] >= start) & (base["dia"] < end)

if use_clasif:
    if sel_clasif:
        mask &= base["clasificacion_producto"].astype(str).isin(sel_clasif)
else:
    if sel_doc:
        mask &= base["doc_prefix"].astype(str).isin(sel_doc)

bf = base[mask].copy()

# En el nuevo esquema: trabajamos siempre con PRIMERA/SEGUNDA (OTROS cuenta como PRIMERA)
bf = bf[bf["calidad"].isin(["PRIMERA", "SEGUNDA"])].copy()

if bf.empty:
    st.warning("Con estos filtros no quedan datos.")
    st.stop()

# Recalcular aggs filtrados
aggs_f = build_aggs(bf)
mensual = aggs_f["mensual"].copy()
diario = aggs_f["diario"].copy()

mensual["mes"] = pd.to_datetime(mensual["mes"], errors="coerce")
diario["dia"] = pd.to_datetime(diario["dia"], errors="coerce")


# ==============================================================================
# 1) Alertas mensuales por %segunda
# ==============================================================================
st.markdown("---")
st.subheader("1) Meses con % SEGUNDA por encima del umbral")

if mensual.empty:
    st.info("No hay agregados mensuales.")
else:
    a = mensual[["anio", "mes", "piezas_total_ps", "pct_segunda"]].copy()
    a = a.sort_values(["anio", "mes"])
    alert_m = a[a["pct_segunda"] >= float(pct_seg_umbral)].copy()
    alert_m["mes"] = alert_m["mes"].dt.strftime("%Y-%m")

    if alert_m.empty:
        st.success(f"No hay meses con %segunda >= {pct_seg_umbral:.1f}%.")
    else:
        st.dataframe(alert_m.sort_values("pct_segunda", ascending=False), use_container_width=True)

        csv_m = alert_m.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV (alertas mensuales)",
            data=csv_m,
            file_name="alertas_mensuales_pct_segunda.csv",
            mime="text/csv",
        )


# ==============================================================================
# 2) Alertas diarias por volumen (z-score rolling)
# ==============================================================================
st.markdown("---")
st.subheader("2) Días atípicos por volumen (picos/caídas)")

if diario.empty:
    st.info("No hay agregados diarios.")
else:
    d = diario[["anio", "dia", "piezas_total_ps", "pct_segunda"]].copy()
    d = d.sort_values("dia").reset_index(drop=True)

    d["roll_mean"] = d["piezas_total_ps"].rolling(window=window, min_periods=max(3, window // 3)).mean()
    d["roll_std"] = d["piezas_total_ps"].rolling(window=window, min_periods=max(3, window // 3)).std()

    d["z"] = (d["piezas_total_ps"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)
    d["z"] = d["z"].replace([np.inf, -np.inf], np.nan).fillna(0)

    d["tipo"] = np.where(d["z"] >= float(z_thr), "PICO", np.where(d["z"] <= -float(z_thr), "CAIDA", "OK"))
    alert_d = d[d["tipo"] != "OK"].copy().sort_values("dia", ascending=False)

    if alert_d.empty:
        st.success(f"No hay días atípicos con |z| >= {z_thr:.1f}.")
    else:
        st.dataframe(
            alert_d[["dia", "piezas_total_ps", "pct_segunda", "z", "tipo"]],
            use_container_width=True,
        )
        csv_d = alert_d.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV (alertas diarias)",
            data=csv_d,
            file_name="alertas_diarias_volumen.csv",
            mime="text/csv",
        )


# ==============================================================================
# 3) Alertas por SKU: %segunda alta con volumen significativo
# ==============================================================================
st.markdown("---")
st.subheader("3) SKUs con % SEGUNDA alto (con filtros anti-ruido)")

tmp = bf.copy()
tmp["is_seg"] = (tmp["calidad"] == "SEGUNDA").astype(int)
tmp["p_seg"] = tmp["piezas"] * tmp["is_seg"]

sku = (
    tmp.groupby("sku", as_index=False)
    .agg(
        piezas_total_ps=("piezas", "sum"),
        piezas_segunda=("p_seg", "sum"),
        dias=("dia", "nunique"),
        first_seen=("dia", "min"),
        last_seen=("dia", "max"),
    )
)
sku["pct_segunda"] = np.where(
    sku["piezas_total_ps"] > 0,
    sku["piezas_segunda"] / sku["piezas_total_ps"] * 100,
    0.0,
)

sku_f = sku[
    (sku["piezas_total_ps"] >= float(min_piezas_ps))
    & (sku["dias"] >= int(min_dias_sku))
    & (sku["pct_segunda"] >= float(sku_pct_thr))
].copy()

sku_f = (
    sku_f.sort_values(["pct_segunda", "piezas_total_ps"], ascending=[False, False])
    .head(int(top_n))
    .reset_index(drop=True)
)

st.caption(
    f"Filtro aplicado: piezas_total_ps >= {min_piezas_ps}, días >= {min_dias_sku}, "
    f"%segunda >= {sku_pct_thr:.1f}%"
)

if sku_f.empty:
    st.success("No hay SKUs que cumplan la condición de alerta con estos umbrales.")
else:
    st.dataframe(sku_f, use_container_width=True)

    csv_s = sku_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV (alertas por SKU)",
        data=csv_s,
        file_name="alertas_sku_pct_segunda.csv",
        mime="text/csv",
    )

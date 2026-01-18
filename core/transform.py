# core/transform.py
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.rules import apply_rules


# ==============================================================================
# Helpers
# ==============================================================================

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _ensure_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza un DF a la estructura base usada por el dashboard.

    Columnas objetivo (mínimas):
      - fecha_dt (datetime) opcional
      - dia (date normalized) obligatorio (derivable)
      - sku (str) obligatorio
      - piezas (num) obligatorio
      - cajas (num) opcional
      - producto (str) opcional
      - documento_ref (str) opcional

    Reglas (derivables vía apply_rules):
      - producto_norm
      - clasificacion_producto: SANSON, POP, GALV_BISA, PH, CF, TG, IMU, OTROS, BISA
      - calidad: PRIMERA / SEGUNDA (OTROS cuenta como PRIMERA)
      - doc_prefix (se mantiene por compatibilidad temporal)
    """
    out = df.copy()

    # ---- renombres típicos (por si entra un DF no-curado) ----
    rename_map = {}
    if "CLAVE" in out.columns and "sku" not in out.columns:
        rename_map["CLAVE"] = "sku"
    if "PIEZAS" in out.columns and "piezas" not in out.columns:
        rename_map["PIEZAS"] = "piezas"
    if "cantidad" in out.columns and "piezas" not in out.columns:
        rename_map["cantidad"] = "piezas"
    if "CAJAS" in out.columns and "cajas" not in out.columns:
        rename_map["CAJAS"] = "cajas"
    if "HORA" in out.columns and "hora" not in out.columns:
        rename_map["HORA"] = "hora"
    if rename_map:
        out = out.rename(columns=rename_map)

    # ---- fechas ----
    if "dia" not in out.columns:
        if "fecha_dt" in out.columns:
            out["dia"] = pd.to_datetime(out["fecha_dt"], errors="coerce").dt.normalize()
        elif "fecha" in out.columns:
            out["dia"] = pd.to_datetime(out["fecha"], errors="coerce").dt.normalize()
        else:
            out["dia"] = pd.NaT
    else:
        out["dia"] = pd.to_datetime(out["dia"], errors="coerce").dt.normalize()

    if "fecha_dt" in out.columns:
        out["fecha_dt"] = pd.to_datetime(out["fecha_dt"], errors="coerce")
    else:
        out["fecha_dt"] = pd.NaT

    # ---- sku / piezas ----
    if "sku" not in out.columns:
        out["sku"] = np.nan
    out["sku"] = out["sku"].astype(str).str.strip()
    out.loc[out["sku"].isin(["", "nan", "None", "NaN"]), "sku"] = np.nan

    if "piezas" not in out.columns:
        out["piezas"] = 0
    out["piezas"] = _safe_num(out["piezas"])

    if "cajas" not in out.columns:
        out["cajas"] = np.nan
    out["cajas"] = pd.to_numeric(out["cajas"], errors="coerce")

    # ---- texto opcional ----
    if "producto" not in out.columns:
        for alt in ["comentario", "NOMBRE  DEL PRODUCTO", "NOMBRE DEL PRODUCTO", "PRODUCTO", "DESCRIPCION"]:
            if alt in out.columns:
                out["producto"] = out[alt].fillna("").astype(str)
                break
        else:
            out["producto"] = ""
    out["producto"] = out["producto"].fillna("").astype(str)

    if "documento_ref" not in out.columns:
        for alt in ["ALMACEN Y/O PEDIDO (SERIE Y FOLIO)", "documento", "doc_ref"]:
            if alt in out.columns:
                out["documento_ref"] = out[alt].fillna("").astype(str)
                break
        else:
            out["documento_ref"] = ""
    out["documento_ref"] = out["documento_ref"].fillna("").astype(str)

    # ---- reglas: asegurar que existan columnas NUEVAS aunque el DF venga "curado" ----
    need_rules = any(
        c not in out.columns
        for c in ["calidad", "doc_prefix", "clasificacion_producto", "producto_norm"]
    )
    if need_rules:
        out = apply_rules(out, producto_col="producto", documento_ref_col="documento_ref")

    # Compat: si por alguna razón quedó OTROS en calidad (datasets viejos), lo pasamos a PRIMERA
    if "calidad" in out.columns:
        out["calidad"] = out["calidad"].fillna("").astype(str).str.strip().str.upper()
        out.loc[out["calidad"] == "OTROS", "calidad"] = "PRIMERA"
        # valores inválidos -> PRIMERA por defecto (seguro)
        out.loc[~out["calidad"].isin(["PRIMERA", "SEGUNDA"]), "calidad"] = "PRIMERA"
    else:
        out["calidad"] = "PRIMERA"

    # Asegurar texto
    if "clasificacion_producto" in out.columns:
        out["clasificacion_producto"] = out["clasificacion_producto"].fillna("").astype(str).str.strip()
    else:
        out["clasificacion_producto"] = ""

    if "producto_norm" in out.columns:
        out["producto_norm"] = out["producto_norm"].fillna("").astype(str)
    else:
        out["producto_norm"] = ""

    if "doc_prefix" in out.columns:
        out["doc_prefix"] = out["doc_prefix"].fillna("").astype(str).str.strip()
    else:
        out["doc_prefix"] = "NA"

    # ---- calendario ----
    out["anio"] = out["dia"].dt.year
    out["mes"] = out["dia"].dt.to_period("M").dt.to_timestamp()
    out["mes_num"] = out["dia"].dt.month

    # ---- hora_ok (si no existe, default False) ----
    if "hora_ok" not in out.columns:
        out["hora_ok"] = False
    out["hora_ok"] = out["hora_ok"].fillna(False).astype(bool)

    # ---- filtros mínimos de calidad de datos ----
    out = out.dropna(subset=["dia", "sku"]).copy()
    out = out[out["piezas"] > 0].copy()

    # ---- orden recomendado ----
    preferred = [
        "fecha_dt", "dia", "anio", "mes", "mes_num", "hora_ok",
        "sku", "piezas", "cajas",
        "calidad", "clasificacion_producto", "doc_prefix",
        "documento_ref", "producto", "producto_norm",
    ]
    preferred = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    out = out[preferred + rest].reset_index(drop=True)

    return out


def _pivot_calidad(df: pd.DataFrame, idx_cols: list[str]) -> pd.DataFrame:
    """
    Genera columnas:
      piezas_primera, piezas_segunda, piezas_otros,
      piezas_total, piezas_total_ps, pct_segunda

    Nota: en el nuevo esquema OTROS ya no se usa como calidad (OTROS cuenta como PRIMERA),
    pero dejamos la columna piezas_otros por compatibilidad (quedará 0).
    """
    g = df.groupby(idx_cols + ["calidad"], as_index=False)["piezas"].sum()

    piv = g.pivot_table(index=idx_cols, columns="calidad", values="piezas", aggfunc="sum", fill_value=0)
    piv = piv.reset_index()

    # asegurar columnas
    for c in ["PRIMERA", "SEGUNDA", "OTROS"]:
        if c not in piv.columns:
            piv[c] = 0.0

    piv = piv.rename(
        columns={
            "PRIMERA": "piezas_primera",
            "SEGUNDA": "piezas_segunda",
            "OTROS": "piezas_otros",
        }
    )

    piv["piezas_total"] = piv["piezas_primera"] + piv["piezas_segunda"] + piv["piezas_otros"]
    piv["piezas_total_ps"] = piv["piezas_primera"] + piv["piezas_segunda"]

    piv["pct_segunda"] = np.where(
        piv["piezas_total_ps"] > 0,
        piv["piezas_segunda"] / piv["piezas_total_ps"] * 100,
        0.0,
    )

    for c in ["piezas_primera", "piezas_segunda", "piezas_otros", "piezas_total", "piezas_total_ps", "pct_segunda"]:
        piv[c] = pd.to_numeric(piv[c], errors="coerce").fillna(0.0)

    return piv


# ==============================================================================
# API pública
# ==============================================================================

def build_aggs(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Retorna:
      - base   : dataframe estandarizado
      - diario : por día (dia/anio)
      - mensual: por mes (mes/anio/mes_num)
      - anual  : por año (anio)
    """
    base = _ensure_base(df)

    diario = _pivot_calidad(base, ["dia", "anio"]).sort_values("dia").reset_index(drop=True)
    mensual = _pivot_calidad(base, ["mes", "anio", "mes_num"]).sort_values(["anio", "mes_num"]).reset_index(drop=True)
    anual = _pivot_calidad(base, ["anio"]).sort_values("anio").reset_index(drop=True)

    return {
        "base": base,
        "diario": diario,
        "mensual": mensual,
        "anual": anual,
    }

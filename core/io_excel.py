# core/io_excel.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from core.rules import apply_rules


# ==============================================================================
# Utilidades de columnas / limpieza
# ==============================================================================

def _canon_col(c: str) -> str:
    s = str(c).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas Unnamed si existen; conserva DF usable."""
    cols = list(df.columns)
    keep = [c for c in cols if not str(c).strip().startswith("Unnamed")]
    if not keep:
        return df.copy()
    out = df[keep].copy()
    if len(out) == 0:
        return out
    out = out.dropna(axis=1, how="all")
    return out


def _apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Renombra columnas usando mapping con match case-insensitive."""
    col_map = {_canon_col(c): c for c in df.columns}
    rename = {}
    for raw, new in mapping.items():
        ck = _canon_col(raw)
        if ck in col_map:
            rename[col_map[ck]] = new
    return df.rename(columns=rename)


def _normalize_sku_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    return out


def _combine_fecha_hora(fecha: pd.Series, hora: pd.Series | None) -> tuple[pd.Series, pd.Series]:
    """
    Devuelve (fecha_dt, hora_ok).
    - fecha_dt: datetime combinando FECHA + HORA si HORA es válida
    - hora_ok: bool si se pudo interpretar la hora
    """
    fecha_norm = pd.to_datetime(fecha, errors="coerce").dt.normalize()

    if hora is None:
        return fecha_norm, pd.Series([False] * len(fecha_norm), index=fecha_norm.index)

    h = hora.copy()
    hora_ok = pd.Series([False] * len(fecha_norm), index=fecha_norm.index)

    # Caso 1: hora como fracción del día (excel) o numérico
    if pd.api.types.is_numeric_dtype(h):
        hn = pd.to_numeric(h, errors="coerce")
        hora_ok = hn.notna()
        td = pd.to_timedelta(hn.fillna(0), unit="D")
        fecha_dt = fecha_norm + td
        return fecha_dt, hora_ok

    # Caso 2: hora como string / datetime
    ht = pd.to_datetime(h, errors="coerce")
    hora_ok = ht.notna()
    td = (
        pd.to_timedelta(ht.dt.hour.fillna(0).astype(int), unit="h")
        + pd.to_timedelta(ht.dt.minute.fillna(0).astype(int), unit="m")
        + pd.to_timedelta(ht.dt.second.fillna(0).astype(int), unit="s")
    )
    fecha_dt = fecha_norm + td
    return fecha_dt, hora_ok


def _motivo_rechazo(df: pd.DataFrame) -> pd.Series:
    motivo = pd.Series([""] * len(df), index=df.index, dtype="object")

    # fecha_dt obligatoria
    if "fecha_dt" in df.columns:
        motivo = np.where(df["fecha_dt"].isna(), "fecha_invalida", motivo)

    # sku obligatorio
    if "sku" in df.columns:
        bad = df["sku"].isna() | (df["sku"].astype(str).str.strip() == "")
        motivo = np.where(bad, (pd.Series(motivo).astype(str) + "|sku_vacio").str.strip("|"), motivo)

    # piezas obligatoria y positiva
    if "piezas" in df.columns:
        p = pd.to_numeric(df["piezas"], errors="coerce")
        motivo = np.where(p.isna(), (pd.Series(motivo).astype(str) + "|piezas_nan").str.strip("|"), motivo)
        motivo = np.where(p.fillna(0) <= 0, (pd.Series(motivo).astype(str) + "|piezas_no_positivas").str.strip("|"), motivo)

    return pd.Series(motivo, index=df.index).astype(str)


# ==============================================================================
# Carga principal
# ==============================================================================

ENTRADAS_MAPPING = {
    "FECHA": "fecha",
    "CLAVE": "sku",
    "PIEZAS": "piezas",
    "HORA": "hora",
    "CAJAS": "cajas",
    "ALMACEN Y/O PEDIDO (SERIE Y FOLIO)": "documento_ref",
    "DOCUMENTO_REF": "documento_ref",
    # variantes de nombre de producto
    "NOMBRE  DEL PRODUCTO": "producto",
    "NOMBRE DEL PRODUCTO": "producto",
    "NOMBRE PRODUCTO": "producto",
    "PRODUCTO": "producto",
    "DESCRIPCION": "producto",
    "DESCRIPCIÓN": "producto",
}


def _read_excel_smart(path: Path, sheet_name: str, dtype_map: dict | None = None) -> pd.DataFrame:
    """
    Intenta leer con skiprows=3 (formato típico). Si no salen columnas útiles,
    reintenta con skiprows=0.
    """
    dtype_map = dtype_map or {}
    df1 = pd.read_excel(path, sheet_name=sheet_name, skiprows=3, dtype=dtype_map)
    df1 = _drop_unnamed(df1)
    df1.columns = [str(c).strip() for c in df1.columns]

    # Si no aparece ninguna columna clave típica, reintenta
    cols_canon = {_canon_col(c) for c in df1.columns}
    looks_ok = any(k in cols_canon for k in ["FECHA", "CLAVE", "PIEZAS", "NOMBRE DEL PRODUCTO", "NOMBRE  DEL PRODUCTO"])
    if looks_ok:
        return df1

    df2 = pd.read_excel(path, sheet_name=sheet_name, skiprows=0, dtype=dtype_map)
    df2 = _drop_unnamed(df2)
    df2.columns = [str(c).strip() for c in df2.columns]
    return df2


def load_entradas_excel(filepath: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee el Excel histórico de entradas:
      - Hoja: "Entrada Inv." si existe; si no, primera hoja
      - Intenta skiprows=3, fallback a skiprows=0
      - Limpia Unnamed / columnas vacías
      - Estandariza y filtra filas no útiles
    Retorna:
      df_ok : dataset limpio + reglas (producto_norm, clasificacion_producto, calidad, doc_prefix)
      df_bad: rechazados con columna 'motivo' (también con reglas para diagnóstico)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path.resolve()}")

    xl = pd.ExcelFile(path)
    sheets = [str(s) for s in xl.sheet_names]

    target_sheet = "Entrada Inv." if "Entrada Inv." in sheets else (sheets[0] if sheets else 0)

    # leer “smart”
    df_raw = _read_excel_smart(path, target_sheet, dtype_map={"CLAVE": str})

    # Renombrar a estándar
    df = _apply_mapping(df_raw, ENTRADAS_MAPPING)

    # Asegurar columnas mínimas (si no, créalas para no romper)
    for c in ["fecha", "sku", "piezas"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalizaciones
    df["sku"] = _normalize_sku_series(df["sku"])
    df["piezas"] = pd.to_numeric(df["piezas"], errors="coerce")

    # cajas opcional
    if "cajas" in df.columns:
        df["cajas"] = pd.to_numeric(df["cajas"], errors="coerce")
    else:
        df["cajas"] = np.nan

    # documento_ref / producto opcional
    if "documento_ref" not in df.columns:
        df["documento_ref"] = ""
    df["documento_ref"] = df["documento_ref"].fillna("").astype(str).str.strip()

    if "producto" not in df.columns:
        # rescate por si vino con un nombre no contemplado
        for alt in ["comentario", "NOMBRE  DEL PRODUCTO", "NOMBRE DEL PRODUCTO", "NOMBRE PRODUCTO", "PRODUCTO", "DESCRIPCION", "DESCRIPCIÓN"]:
            if alt in df_raw.columns:
                df["producto"] = df_raw[alt].fillna("").astype(str)
                break
        else:
            df["producto"] = ""
    df["producto"] = df["producto"].fillna("").astype(str)

    # combinar fecha + hora (si existe)
    if "hora" in df.columns:
        fecha_dt, hora_ok = _combine_fecha_hora(df["fecha"], df["hora"])
    else:
        fecha_dt, hora_ok = _combine_fecha_hora(df["fecha"], None)

    df["fecha_dt"] = pd.to_datetime(fecha_dt, errors="coerce")
    df["hora_ok"] = hora_ok.astype(bool)
    df["dia"] = df["fecha_dt"].dt.normalize()

    # Motivos de rechazo
    motivo = _motivo_rechazo(df)
    df_bad = df[motivo != ""].copy()
    if len(df_bad):
        df_bad["motivo"] = motivo[motivo != ""]

    df_ok = df[motivo == ""].copy()

    # Reglas de negocio (nuevas columnas incluidas)
    df_ok = apply_rules(df_ok, producto_col="producto", documento_ref_col="documento_ref")

    # Orden recomendado de columnas
    cols_first = [
        "fecha_dt", "dia", "hora_ok",
        "sku", "piezas", "cajas",
        "calidad", "clasificacion_producto",  # NUEVO
        "doc_prefix",                         # compat temporal
        "documento_ref", "producto", "producto_norm",  # NUEVO
    ]
    cols_first = [c for c in cols_first if c in df_ok.columns]
    cols_rest = [c for c in df_ok.columns if c not in cols_first]
    df_ok = df_ok[cols_first + cols_rest].reset_index(drop=True)

    # Para rechazados, también aplica reglas (útil para diagnóstico)
    if len(df_bad):
        df_bad = apply_rules(df_bad, producto_col="producto", documento_ref_col="documento_ref")
        if "motivo" in df_bad.columns:
            cols = [c for c in df_bad.columns if c != "motivo"] + ["motivo"]
            df_bad = df_bad[cols].reset_index(drop=True)

    return df_ok, df_bad

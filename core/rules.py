# core/rules.py
from __future__ import annotations

import re
import unicodedata
import pandas as pd


# ==============================================================================
# Validaciones
# ==============================================================================

def require_columns(df: pd.DataFrame, cols: list[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        loc = f" ({where})" if where else ""
        raise ValueError(
            f"Faltan columnas{loc}: {missing}. "
            f"Disponibles: {list(df.columns)}"
        )


# ==============================================================================
# Normalización de texto
# ==============================================================================

def _strip_accents(s: str) -> str:
    # Quita acentos para hacer matching más robusto
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).replace("\u00A0", " ").strip()
    s = _strip_accents(s)
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _token_re(token: str) -> re.Pattern:
    """
    Match de token completo, evitando falsos positivos:
      - "POP" NO debe matchear "POPOTE"
      - "BISA" NO debe matchear "BISAGRA"
    """
    esc = re.escape(token.upper())
    return re.compile(rf"(?:^|[^A-Z0-9]){esc}(?:[^A-Z0-9]|$)")


# ==============================================================================
# Clasificación por NOMBRE PRODUCTO (sustituye doc_prefix)
# ==============================================================================

# Orden deseado en UI / filtros
CLASIFICACION_ORDER: list[str] = [
    "SANSON", "POP", "GALV_BISA", "PH", "CF", "TG", "IMU", "OTROS", "BISA"
]

RE_SANSON = _token_re("SANSON")
RE_POP = _token_re("POP")
RE_BISA = _token_re("BISA")
RE_IMU = _token_re("IMU")
RE_IMUSA = _token_re("IMUSA")

RE_PH = _token_re("PH")
RE_CF = _token_re("CF")
RE_TG = _token_re("TG")


def classify_producto(producto_text: str) -> str:
    """
    Regla (prioridad alta -> baja), basada en NOMBRE PRODUCTO:

      1) SANSON -> SANSON
      2) POP -> POP
      3) PH -> PH
      4) CF -> CF
      5) TG -> TG
      6) IMU / IMUSA -> IMU
      7) BISA + GALV* -> GALV_BISA
      8) BISA -> BISA
      9) default -> OTROS

    Nota: OTROS cuenta como PRIMERA en KPIs (decisión confirmada).
    """
    t = normalize_text(producto_text)
    if not t:
        return "OTROS"

    # SEGUNDA (mandan)
    if RE_SANSON.search(t):
        return "SANSON"
    if RE_POP.search(t):
        return "POP"

    # Clientes / etiquetas cortas (token completo)
    if RE_PH.search(t):
        return "PH"
    if RE_CF.search(t):
        return "CF"
    if RE_TG.search(t):
        return "TG"

    # IMU / IMUSA
    if RE_IMUSA.search(t) or RE_IMU.search(t):
        return "IMU"

    # GALV_BISA
    has_bisa = RE_BISA.search(t) is not None
    has_galv = ("GALV" in t)  # cubre GALV, GALVANIZADO, GALVANIZADA, etc.
    if has_bisa and has_galv:
        return "GALV_BISA"

    # BISA genérico
    if has_bisa:
        return "BISA"

    return "OTROS"


def calidad_from_clasificacion(clasificacion: str) -> str:
    """
    Calidad (KPIs y mix):
      - SEGUNDA: SANSON, POP
      - PRIMERA: el resto (incluye OTROS)
    """
    c = (clasificacion or "").strip().upper()
    if c in {"SANSON", "POP"}:
        return "SEGUNDA"
    return "PRIMERA"


def classify_calidad(producto_text: str) -> str:
    """
    Backward compatible: si en algún lugar llaman classify_calidad(producto),
    ahora se deriva de la nueva clasificación.
    """
    clas = classify_producto(producto_text)
    return calidad_from_clasificacion(clas)


# ==============================================================================
# doc_prefix desde documento_ref (se mantiene temporalmente por compatibilidad)
# ==============================================================================

def extract_doc_prefix(documento_ref) -> str:
    """
    Extrae un prefijo estable para agrupar (ej: STOCK, VT, BO, etc.)
    Estrategia:
      1) Normaliza texto
      2) Toma primer token (o lo que esté antes de '-')
      3) Se queda con letras iniciales (y algunos casos especiales)
    """
    s = normalize_text(documento_ref)
    if not s:
        return "NA"

    # corta antes de guion si existe
    left = s.split("-", 1)[0].strip()

    # primer token por espacio
    tok = left.split(" ", 1)[0].strip()
    if not tok:
        return "NA"

    # casos comunes
    if tok.startswith("STOCK"):
        return "STOCK"
    if tok.startswith("VT"):
        return "VT"
    if tok.startswith("BO"):
        return "BO"
    if tok.startswith("PO"):
        return "PO"
    if tok.startswith("OC"):
        return "OC"

    # letras iniciales del token (ej: VT1 -> VT)
    m = re.match(r"^([A-Z]+)", tok)
    if m:
        pref = m.group(1)
        # limita tamaño para no crear categorías raras
        return pref[:10] if pref else "OTROS_DOC"

    return "OTROS_DOC"


# ==============================================================================
# Aplicación de reglas al DF
# ==============================================================================

def apply_rules(
    df: pd.DataFrame,
    producto_col: str = "producto",
    documento_ref_col: str = "documento_ref",
) -> pd.DataFrame:
    """
    Añade:
      - producto_norm
      - clasificacion_producto (SANSON/POP/GALV_BISA/PH/CF/TG/IMU/OTROS/BISA)
      - calidad (PRIMERA/SEGUNDA) [OTROS cuenta como PRIMERA]
      - doc_prefix (compat temporal)

    No modifica columnas base; devuelve copia.
    """
    out = df.copy()

    if producto_col not in out.columns:
        # fallback típico si el excel usa otro nombre
        for alt in ["comentario", "NOMBRE  DEL PRODUCTO", "NOMBRE DEL PRODUCTO", "PRODUCTO", "DESCRIPCION"]:
            if alt in out.columns:
                producto_col = alt
                break

    if documento_ref_col not in out.columns:
        for alt in ["ALMACEN Y/O PEDIDO (SERIE Y FOLIO)", "documento", "doc_ref"]:
            if alt in out.columns:
                documento_ref_col = alt
                break

    # crea columnas aunque falten (para no romper pipeline)
    if producto_col not in out.columns:
        out["producto_norm"] = ""
        out["clasificacion_producto"] = "OTROS"
        out["calidad"] = "PRIMERA"
    else:
        out["producto_norm"] = out[producto_col].apply(normalize_text)
        out["clasificacion_producto"] = out["producto_norm"].apply(classify_producto)
        out["calidad"] = out["clasificacion_producto"].apply(calidad_from_clasificacion)

    if documento_ref_col in out.columns:
        out["doc_prefix"] = out[documento_ref_col].apply(extract_doc_prefix)
    else:
        out["doc_prefix"] = "NA"

    return out

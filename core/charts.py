# core/charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _hd_fig(w: float = 12.0, h: float = 5.0, dpi: int = 220):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    return fig, ax


def _clean_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _fmt_pct(v: float) -> str:
    s = f"{float(v):.1f}"
    s = s.rstrip("0").rstrip(".")
    return f"{s}%"


def _fmt_int(v: float) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return ""


def _annotate_stacked(
    ax,
    bars,
    values,
    bottoms=None,
    *,
    fmt="int",
    min_height: float = 0.0,
    fontsize: int = 8,
):
    """Anota etiquetas centradas en cada segmento de una barra apilada.

    - bars: BarContainer (resultado de ax.bar)
    - values: alturas (1D)
    - bottoms: base inferior de cada barra (1D) o None (0)

    Estrategia:
      - Si el segmento es suficientemente alto, etiqueta centrada.
      - Si es muy pequeño, etiqueta arriba del segmento.
    """
    if bottoms is None:
        bottoms = np.zeros(len(values))

    # Umbral de legibilidad (para piezas) en función del rango Y actual
    try:
        y0, y1 = ax.get_ylim()
        y_span = max(1.0, float(y1) - float(y0))
    except Exception:
        y_span = 1.0

    for rect, h, b in zip(bars.patches, values, bottoms):
        if h is None:
            continue
        try:
            h = float(h)
            b = float(b)
        except Exception:
            continue

        if h <= 0:
            continue
        if h < float(min_height):
            continue

        if fmt == "pct":
            label = _fmt_pct(h)
        else:
            label = _fmt_int(h)

        if not label:
            continue

        x = rect.get_x() + rect.get_width() / 2.0
        y_center = b + h / 2.0

        # si es muy pequeño, coloca arriba para que se lea
        if fmt == "pct":
            too_small = h < 6.0
        else:
            too_small = h < 0.055 * y_span

        if too_small:
            y = b + h + (0.8 if fmt == "pct" else 0.02 * y_span)
            va = "bottom"
        else:
            y = y_center
            va = "center"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
            clip_on=False,
        )


# ==============================================================================
# 1) Anual en % (Mix de calidad)
# ==============================================================================

def chart_anual_pct(anual: pd.DataFrame):
    """Mix anual por calidad (en %).

    anual debe tener:
      - anio
      - piezas_primera
      - piezas_segunda

    Cambios solicitados:
      - Leyenda fuera del gráfico
      - Etiquetas con % en cada segmento
    """
    df = anual.copy()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["anio"]).sort_values("anio")

    p1 = pd.to_numeric(df.get("piezas_primera", 0), errors="coerce").fillna(0)
    p2 = pd.to_numeric(df.get("piezas_segunda", 0), errors="coerce").fillna(0)

    ps = p1 + p2
    pct_prim = np.where(ps > 0, p1 / ps * 100.0, 0.0)
    pct_seg = np.where(ps > 0, p2 / ps * 100.0, 0.0)

    fig, ax = _hd_fig(11.5, 4.8)

    x = df["anio"].astype(int).to_numpy()
    bars1 = ax.bar(x, pct_prim, label="PRIMERA")
    bars2 = ax.bar(x, pct_seg, bottom=pct_prim, label="SEGUNDA")

    ax.set_ylabel("% del total (PRIMERA+SEGUNDA)")
    ax.set_title("Mix anual por calidad (en %)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x.tolist())

    # Etiquetas por segmento
    _annotate_stacked(ax, bars1, pct_prim, bottoms=np.zeros_like(pct_prim), fmt="pct", fontsize=8)
    _annotate_stacked(ax, bars2, pct_seg, bottoms=pct_prim, fmt="pct", fontsize=8)

    # Leyenda fuera para no tapar
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    _clean_axes(ax)

    # Deja espacio para la leyenda a la derecha
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    return fig


# ==============================================================================
# 2) Mensual stacked (PRIMERA vs SEGUNDA)
# ==============================================================================

def chart_mensual_stacked(mensual: pd.DataFrame, year: int):
    """Producción mensual (piezas) PRIMERA vs SEGUNDA.

    Cambios solicitados:
      - Barras más anchas
      - Etiquetas en cada segmento (piezas)
    """
    df = mensual.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    df = df.dropna(subset=["mes"]).sort_values("mes")

    p1 = pd.to_numeric(df.get("piezas_primera", 0), errors="coerce").fillna(0)
    p2 = pd.to_numeric(df.get("piezas_segunda", 0), errors="coerce").fillna(0)

    fig, ax = _hd_fig(11.5, 5.2)

    # Ancho en días (matplotlib usa "days" como unidad para fechas)
    bar_width_days = 25

    bars1 = ax.bar(df["mes"], p1, width=bar_width_days, label="PRIMERA")
    bars2 = ax.bar(df["mes"], p2, bottom=p1, width=bar_width_days, label="SEGUNDA")

    ax.set_ylabel("Piezas")
    ax.set_title(f"Producción mensual {year}: PRIMERA vs SEGUNDA")
    ax.legend()

    # margen extra arriba para evitar que se corten etiquetas
    total = (p1 + p2).to_numpy()
    ymax = float(np.nanmax(total)) if len(total) else 0.0
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.15)

    # Etiquetas por segmento (piezas) - después de fijar YLIM
    _annotate_stacked(ax, bars1, p1.to_numpy(), bottoms=np.zeros(len(p1)), fmt="int", fontsize=8)
    _annotate_stacked(ax, bars2, p2.to_numpy(), bottoms=p1.to_numpy(), fmt="int", fontsize=8)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig


# ==============================================================================
# 3) % Segunda mensual
# ==============================================================================

def chart_pct_segunda_mensual(mensual: pd.DataFrame, year: int):
    df = mensual.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    df = df.dropna(subset=["mes"]).sort_values("mes")

    fig, ax = _hd_fig(11.5, 4.6)
    ax.plot(df["mes"], df["pct_segunda"])

    ax.set_ylabel("% Segunda (sobre PRIMERA+SEGUNDA)")
    ax.set_title(f"% Segunda mensual {year}")

    ymax = max(5.0, float(df["pct_segunda"].max()) * 1.15) if len(df) else 10.0
    ax.set_ylim(0, ymax)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig


# ==============================================================================
# 4) Diario line (total PRIMERA+SEGUNDA)
# ==============================================================================

def chart_diario_line(diario: pd.DataFrame, year: int):
    """Producción diaria total (PRIMERA+SEGUNDA)."""
    df = diario.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["dia"] = pd.to_datetime(df["dia"], errors="coerce")
    df = df.dropna(subset=["dia"]).sort_values("dia")

    fig, ax = _hd_fig(11.5, 4.8)
    ax.plot(df["dia"], df["piezas_total_ps"])

    ax.set_ylabel("Piezas (PRIMERA+SEGUNDA)")
    ax.set_title(f"Producción diaria {year}")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig

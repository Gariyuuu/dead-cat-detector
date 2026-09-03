"""Publication figure styling and builders.

Colour follows the job it does, not taste:

* **categorical** (crash type, VIX regime) uses a fixed slot order - blue,
  orange, aqua - validated all-pairs for colour-vision deficiency
  (worst CVD dE 9.2, worst normal-vision dE 24.0). Slots are assigned to
  entities and never cycled or re-assigned when a subset is plotted.
* **sequential** (counts, AUC) uses a single blue ramp, light to dark.
* **diverging** (CAR, which has a meaningful zero) uses blue-gray-red with a
  neutral midpoint anchored at zero.

Aqua sits below 3:1 against the light surface, so every categorical figure
ships direct labels as relief rather than relying on the legend alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#3987e5", "#9ec5f4", "#f0efec",
             "#f6b6b5", "#e34948", "#b8302f", "#8a1f1e"]

CMAP_SEQ = LinearSegmentedColormap.from_list("dc_seq", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("dc_div", DIVERGING)

# Fixed entity -> slot assignment. Colour follows the entity, never its rank.
CRASH_TYPE_COLORS = {
    "broad_market": CATEGORICAL[0],
    "sector": CATEGORICAL[1],
    "idiosyncratic": CATEGORICAL[2],
}
CRASH_TYPE_LABELS = {
    "broad_market": "Broad-market",
    "sector": "Sector",
    "idiosyncratic": "Idiosyncratic",
}
REGIME_COLORS = {"high_vix": CATEGORICAL[0], "low_vix": CATEGORICAL[1]}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "medium",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def finish(fig, ax, title: str, subtitle: str | None = None, source: str | None = None):
    """Title block + source note in text ink, never series colour."""
    if subtitle:
        # Offset points, not axes fractions: an axes-fraction offset scales with
        # figure height and collides on tall figures.
        ax.set_title(title, loc="left", color=INK, pad=26)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=9.5, color=INK_2, va="bottom", ha="left")
    else:
        ax.set_title(title, loc="left", color=INK)
    if source:
        fig.text(0.0, -0.045, source, fontsize=8, color=INK_MUTED, ha="left", va="top")
    return fig


def save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def pct_axis(ax, axis: str = "y", decimals: int = 1):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def band(ax, x, mean, lo, hi, color, label, direct_label=True, lw=2.0, z=3):
    """A CAR path with its bootstrap band and an optional end-of-line label."""
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=z - 1)
    ax.plot(x, mean, color=color, linewidth=lw, label=label, zorder=z,
            solid_capstyle="round")
    if direct_label:
        ax.annotate(label, xy=(x[-1], mean[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", ha="left", fontweight="medium")
    return ax


def zero_line(ax):
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)


def heatmap(ax, mat: pd.DataFrame, cmap, norm=None, fmt="{:+.2%}", cbar_label="",
            annotate=True, text_threshold=0.62, cbar=True):
    """Cell grid with a 2px surface gap between cells and readable annotations."""
    im = ax.imshow(mat.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    if annotate:
        vals = mat.to_numpy(dtype=float)
        rgba = im.cmap(im.norm(vals))
        lum = 0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if not np.isfinite(vals[i, j]):
                    continue
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center",
                        fontsize=8.5,
                        color="#ffffff" if lum[i, j] < text_threshold else INK)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=GRID)
        cb.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.1f}%"))
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8.5, color=INK_2)
    return im


def diverging_norm(values, center: float = 0.0) -> TwoSlopeNorm:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    m = max(abs(np.nanmin(v) - center), abs(np.nanmax(v) - center), 1e-9)
    return TwoSlopeNorm(vmin=center - m, vcenter=center, vmax=center + m)

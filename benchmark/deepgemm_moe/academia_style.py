"""Academic matplotlib style setup.

Import this module or paste its contents into a script / notebook to get:

- Journal-ready rcParams (Arial Bold, RColorBrewer "Paired" palette, ...)
- ``fp`` — a ``FontProperties`` object (Arial Bold 13) for explicit use
- ``PAIRED``, ``MARKERS``, ``LINESTYLES`` — convenience constants
- ``style_ax(ax)`` / ``style_fig(fig, ...)`` — apply ``fp`` to all text and
  optionally create a top or inside legend in one call

Requires: matplotlib, cycler. For Arial on Ubuntu run install-fonts.sh.
"""

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager

# -- Academic style (applied via rcParams) -----------------------------------
_ACADEMIC_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.weight": "bold",
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.prop_cycle": cycler("color", [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
        "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928",
    ]),
    "lines.linewidth": 1.5,
    "lines.markersize": 12,
    "axes.linewidth": 1.5,
    "axes.edgecolor": "black",
    "axes.axisbelow": True,
    "xtick.major.width": 1.5,
    "xtick.major.size": 3,
    "ytick.major.width": 1.5,
    "ytick.major.size": 3,
    "xtick.minor.width": 1.0,
    "xtick.minor.size": 2,
    "ytick.minor.width": 1.0,
    "ytick.minor.size": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 1.0,
    "grid.alpha": 0.6,
    "axes.grid.axis": "y",
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "hatch.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}
matplotlib.rcParams.update(_ACADEMIC_RC)

# -- Font (Arial Bold 13) ----------------------------------------------------
fp_path = font_manager.findfont(font_manager.FontProperties(family="Arial"))
fp = font_manager.FontProperties(fname=fp_path, weight="bold", size=13)

# -- RColorBrewer "Paired" -- C0-C11 ----------------------------------------
BLUE_LT = "#A6CEE3"
BLUE_DK = "#1F78B4"
GREEN_LT = "#B2DF8A"
GREEN_DK = "#33A02C"
RED_LT = "#FB9A99"
RED_DK = "#E31A1C"
ORANGE_LT = "#FDBF6F"
ORANGE_DK = "#FF7F00"
PURPLE_LT = "#CAB2D6"
PURPLE_DK = "#6A3D9A"
YELLOW_LT = "#FFFF99"
BROWN_DK = "#B15928"
PAIRED = [
    BLUE_LT,
    BLUE_DK,
    GREEN_LT,
    GREEN_DK,
    RED_LT,
    RED_DK,
    ORANGE_LT,
    ORANGE_DK,
    PURPLE_LT,
    PURPLE_DK,
    YELLOW_LT,
    BROWN_DK,
]

MARKERS = ["s", "D", "^", "d", "o", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def style_ax(ax, _fp=None, enforce=False):
    """Apply fp (Arial Bold 13) to all text elements on a single Axes.

    Parameters
    ----------
    enforce : bool, default False
        When True, also re-apply all academic rcParams (ticks, grid, spines,
        line widths, marker sizes) directly onto the axes.
    """
    _fp = _fp or fp
    rc = _ACADEMIC_RC

    ax.xaxis.label.set_fontproperties(_fp)
    ax.yaxis.label.set_fontproperties(_fp)
    ax.title.set_fontproperties(_fp)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(_fp)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontproperties(_fp)

    if not enforce:
        return

    for spine in ax.spines.values():
        spine.set_linewidth(rc["axes.linewidth"])
        spine.set_edgecolor(rc["axes.edgecolor"])

    ax.tick_params(
        axis="both",
        which="major",
        direction=rc["xtick.direction"],
        width=rc["xtick.major.width"],
        length=rc["xtick.major.size"],
        labelsize=rc["xtick.labelsize"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction=rc["xtick.direction"],
        width=rc["xtick.minor.width"],
        length=rc["xtick.minor.size"],
    )

    ax.set_axisbelow(rc["axes.axisbelow"])
    grid_axis = rc["axes.grid.axis"]
    grid_kw = {
        "linestyle": rc["grid.linestyle"],
        "linewidth": rc["grid.linewidth"],
        "alpha": rc["grid.alpha"],
    }
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, **grid_kw)
    else:
        ax.xaxis.grid(False)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, **grid_kw)
    else:
        ax.yaxis.grid(False)

    for line in ax.get_lines():
        line.set_linewidth(rc["lines.linewidth"])
        line.set_markersize(rc["lines.markersize"])


def style_fig(
    fig,
    _fp=None,
    legend_ncol=None,
    legend_level="fig",
    legend_loc="top",
    enforce=False,
    **legend_kw,
):
    """Apply fp to every Axes in fig and optionally create a legend."""
    _fp = _fp or fp
    if enforce:
        matplotlib.rcParams.update(_ACADEMIC_RC)
    for ax in fig.get_axes():
        style_ax(ax, _fp, enforce=enforce)

    if legend_ncol is None:
        for leg in fig.legends:
            for t in leg.get_texts():
                t.set_fontproperties(_fp)
        return

    if legend_level == "fig":
        seen = {}
        for ax in fig.get_axes():
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h

        for ax in fig.get_axes():
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        if seen:
            if legend_loc == "top":
                kw = {
                    "ncol": legend_ncol,
                    "frameon": False,
                    "prop": _fp,
                    "loc": "upper center",
                    "bbox_to_anchor": (0.5, 1.0),
                }
            else:
                kw = {
                    "ncol": legend_ncol,
                    "frameon": False,
                    "prop": _fp,
                    "loc": legend_loc,
                }
            kw.update(legend_kw)
            fig.legend(list(seen.values()), list(seen.keys()), **kw)

    elif legend_level == "ax":
        for ax in fig.get_axes():
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                continue
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            if legend_loc == "top":
                kw = {
                    "ncol": legend_ncol,
                    "frameon": False,
                    "prop": _fp,
                    "loc": "lower center",
                    "bbox_to_anchor": (0.5, 1.0),
                }
            else:
                kw = {
                    "ncol": legend_ncol,
                    "frameon": False,
                    "prop": _fp,
                    "loc": legend_loc,
                }
            kw.update(legend_kw)
            ax.legend(handles, labels, **kw)

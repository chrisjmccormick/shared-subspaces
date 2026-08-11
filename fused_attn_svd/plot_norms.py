"""Plots for the weight-RMS survey.

Reads the CSV/NPZ written by norm_survey.py and produces three figures:

  fig1_rms_pinned.png       per-matrix RMS / (0.5/sqrt(d)) vs layer, Glimmer | Qwen3-8B.
                            Shared log y-axis, so the two models are directly comparable:
                            Glimmer collapses onto 1.0, Qwen3 scatters over 2-4.
  fig2_per_head_spread.png  per-head RMS max/min within each matrix, vs layer. This is the
                            thing the aggregate statistic is structurally blind to
                            (RMS(W)^2 is exactly the mean of the per-head RMS^2).
  fig3_gate_head_band.png   full per-head RMS spread for the widest-spread matrix, as a
                            min-max band + median vs layer.

Usage:
    python plot_norms.py --results-dir results --out-dir .
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Categorical palette, fixed slot order (validated for the adjacent pairlist used by
# line charts). Colour follows the matrix type, never its rank, so the same matrix is
# the same hue in every panel of every figure.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#4a3aa7", "#e34948"]
COLOR = {
    "self_attn.q_proj":    SLOTS[0],
    "self_attn.k_proj":    SLOTS[1],
    "self_attn.v_proj":    SLOTS[2],
    "self_attn.o_proj":    SLOTS[3],
    "self_attn.gate_proj": SLOTS[4],
    "mlp.gate_proj":       SLOTS[5],
    "mlp.up_proj":         SLOTS[6],
    "mlp.down_proj":       SLOTS[7],
}
LABEL = {k: k.replace("self_attn.", "attn ").replace("mlp.", "mlp ").replace("_proj", "")
         for k in COLOR}
HEAD_MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "self_attn.gate_proj"]

INK = "#1a1a19"
INK_MUTED = "#6b6a66"
GRID = "#e3e2df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(results_dir: Path, model: str) -> tuple[dict, dict]:
    """-> (rows keyed by (layer, matrix), per-head arrays keyed 'layer|matrix')."""
    rows: dict[tuple[int, str], dict] = {}
    with open(results_dir / f"{model}_norms.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(int(r["layer"]), r["matrix"])] = {
                k: (float(v) if k not in ("matrix", "shape") else v) for k, v in r.items()
            }
    npz_path = results_dir / f"{model}_norms.npz"
    heads = dict(np.load(npz_path)) if npz_path.exists() else {}
    return rows, heads


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Layers are integers; matplotlib's default locator invents 0.5 steps on short runs.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def plain_log_ticks(ax) -> None:
    """Log axis with readable labels -- the default renders '1.00012 x 10^0'."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.3g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def series(rows: dict, matrix: str, field: str) -> tuple[list[int], list[float]]:
    pts = sorted((L, r[field]) for (L, m), r in rows.items() if m == matrix)
    pts = [(L, v) for L, v in pts if not math.isnan(v)]
    return [p[0] for p in pts], [p[1] for p in pts]


def label_ends(ax, items: list[tuple[float, str, str]], min_gap: float = 0.045) -> None:
    """Direct-label series at the right edge, pushed apart so they stay readable.

    The validator WARNs that aqua/yellow/magenta fall below 3:1 on the light surface,
    and that warning is not dismissable -- identity for those series must not rest on
    the swatch alone. `items` is [(y_data, text, color)]; call AFTER limits are final.

    Positions are solved in axes fraction: sort by y, then sweep upward enforcing a
    minimum gap, then sweep down from the top so the stack stays inside the axes.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"

    def to_frac(y: float) -> float:
        if log:
            return (math.log10(y) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (y - lo) / (hi - lo)

    placed = sorted(((to_frac(y), t, c) for y, t, c in items), key=lambda z: z[0])
    ys = [p[0] for p in placed]
    for i in range(1, len(ys)):                       # push up
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - 1.0
    if overflow > 0:                                  # pull the stack back inside
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)

    for y_frac, (_, text, color) in zip(ys, placed):
        ax.annotate(text, xy=(1.01, y_frac), xycoords="axes fraction",
                    fontsize=7.5, color=color, va="center", ha="left",
                    annotation_clip=False)


def fig1(data: dict, out: Path) -> None:
    """Aggregate RMS relative to 0.5/sqrt(d), both models, shared log y."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    spreads = {}
    for ax, (model, title) in zip(axes, [("glimmer", "Muse Glimmer 30B"),
                                         ("qwen3-8b", "Qwen3-8B")]):
        rows, _ = data[model]
        ends, allv = [], []
        for matrix in COLOR:
            xs, ys = series(rows, matrix, "ratio_to_target")
            if not xs:
                continue
            ax.plot(xs, ys, lw=2.0, color=COLOR[matrix], label=LABEL[matrix],
                    solid_capstyle="round")
            ends.append((ys[-1], LABEL[matrix], COLOR[matrix]))
            allv.extend(ys)
        spreads[model] = (min(allv), max(allv))
        ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        style_axes(ax, title, "layer", "")
        # Glimmer's eight series lie on top of each other, so end labels would be a
        # pile of overlapping text claiming a distinction the data does not have.
        # Say that instead; label directly only where the series actually separate.
        if max(allv) / min(allv) < 1.01:
            ax.annotate(f"all 8 matrix types coincide\n"
                        f"(spread {100*(max(allv)/min(allv)-1):.3f}% across "
                        f"{len(allv)} matrices)",
                        xy=(0.5, 0.42), xycoords="axes fraction", ha="center",
                        fontsize=8.5, color=INK_MUTED)
        else:
            label_ends(ax, ends)
    top = max(hi for _, hi in spreads.values())
    axes[0].set_ylim(0.9, top * 1.10)          # shared: set once, after both panels
    axes[0].set_ylabel(r"weight RMS $\div\ 0.5/\sqrt{d_{model}}$", color=INK_MUTED)
    axes[1].annotate("target = 1.0", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
                     xytext=(0, 5), textcoords="offset points", ha="right",
                     fontsize=8, color=INK_MUTED)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.suptitle("Per-matrix weight RMS is pinned to $0.5/\\sqrt{d}$ in Glimmer, "
                 "free in Qwen3-8B", x=0.5, y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_rms_pinned.png", bbox_inches="tight")
    plt.close(fig)


def fig2(data: dict, out: Path) -> None:
    """Per-head RMS max/min -- what the aggregate cannot see."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, (model, title) in zip(axes, [("glimmer", "Muse Glimmer 30B"),
                                         ("qwen3-8b", "Qwen3-8B")]):
        rows, _ = data[model]
        ends = []
        for matrix in HEAD_MATS:
            xs, ys = series(rows, matrix, "head_max_over_min")
            if not xs:
                continue
            ax.plot(xs, ys, lw=2.0, color=COLOR[matrix], label=LABEL[matrix],
                    solid_capstyle="round")
            ends.append((ys[-1], LABEL[matrix], COLOR[matrix]))
        ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        style_axes(ax, title, "layer", "")
        label_ends(ax, ends)
    axes[0].set_ylabel("per-head RMS  max / min", color=INK_MUTED)
    for ax in axes:
        ax.set_ylim(bottom=0.90)
    axes[0].annotate("1.0 = every head identical", xy=(0.02, 1.0),
                     xycoords=("axes fraction", "data"), xytext=(0, -13),
                     textcoords="offset points", fontsize=8, color=INK_MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Spread of per-head weight RMS within each attention matrix, by layer",
                 x=0.5, y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_per_head_spread.png", bbox_inches="tight")
    plt.close(fig)


def fig3(data: dict, out: Path) -> None:
    """Full per-head spread for the widest matrix in each model: band + median."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (model, title) in zip(axes, [("glimmer", "Muse Glimmer 30B"),
                                         ("qwen3-8b", "Qwen3-8B")]):
        rows, heads = data[model]
        if not heads:
            ax.text(0.5, 0.5, "no per-head arrays (.npz missing)", ha="center",
                    va="center", transform=ax.transAxes, color=INK_MUTED)
            continue
        # widest-spread head-bearing matrix, by median max/min across layers
        best, best_spread = None, 0.0
        for matrix in HEAD_MATS:
            _, ys = series(rows, matrix, "head_max_over_min")
            if ys and float(np.median(ys)) > best_spread:
                best, best_spread = matrix, float(np.median(ys))
        layers, lo, hi, med = [], [], [], []
        for key in sorted((k for k in heads if k.endswith(f"|{best}|head")),
                          key=lambda k: int(k.split("|")[0])):
            h = heads[key]
            layers.append(int(key.split("|")[0]))
            lo.append(h.min()); hi.append(h.max()); med.append(float(np.median(h)))
        c = COLOR[best]
        ax.fill_between(layers, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(layers, med, lw=2.0, color=c, label=f"{LABEL[best]} — median head")
        ax.plot(layers, lo, lw=1.0, color=c, alpha=0.55)
        ax.plot(layers, hi, lw=1.0, color=c, alpha=0.55)
        style_axes(ax, f"{title} — {LABEL[best]}  (median max/min {best_spread:.2f}×)",
                   "layer", "per-head weight RMS")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Per-head weight RMS: band spans min→max across heads in the layer",
                 x=0.5, y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_gate_head_band.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    rd, od = Path(args.results_dir), Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    data = {}
    for model in ("glimmer", "qwen3-8b"):
        try:
            data[model] = load(rd, model)
        except FileNotFoundError as e:
            raise SystemExit(f"missing results for {model}: {e}")

    fig1(data, od)
    fig2(data, od)
    fig3(data, od)
    print(f"wrote fig1/fig2/fig3 -> {od}")


if __name__ == "__main__":
    main()

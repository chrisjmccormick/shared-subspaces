"""Figures for the attention-gate activation statistics (gate_stats.py output).

  fig8_gate_openness.png   (A) per-layer distribution of per-head mean gate
                           openness on calibration text -- the runtime
                           suppression picture. (B) the weights->runtime join:
                           per-head raw gate_proj RMS vs measured openness.

Reads results/glimmer_gate_stats.csv and results/glimmer_sigmas.npz.

Usage:
    python plot_gate_stats.py --results-dir results --out-dir .
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sps

from plot_fused_ranks import SEQ_BLUE
from plot_norms import GRID, INK_MUTED, label_ends, style_axes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results_dir / "glimmer_gate_stats.csv",
                                    encoding="utf-8")))
    n_layers = max(int(r["layer"]) for r in rows) + 1
    nH = max(int(r["head"]) for r in rows) + 1
    openness = np.zeros((n_layers, nH))
    closed = np.zeros((n_layers, nH))
    for r in rows:
        openness[int(r["layer"]), int(r["head"])] = float(r["openness"])
        closed[int(r["layer"]), int(r["head"])] = float(r["frac_closed"])

    z = dict(np.load(args.results_dir / "glimmer_sigmas.npz"))
    g_rms = np.stack([z[f"L{L}|rms_head|gate_proj"] for L in range(n_layers)])

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1],
                                          "wspace": 0.42})

    # (A) heads sorted by openness within each layer -> distribution by depth.
    ax = axes[0]
    im = ax.imshow(np.sort(openness, axis=1).T, aspect="auto", origin="lower",
                   cmap=cmap, vmin=0, vmax=1,
                   extent=(-0.5, n_layers - 0.5, -0.5, nH - 0.5))
    style_axes(ax, "Most heads are mostly closed on real text",
               "layer", "heads (sorted by openness within layer)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean gate openness", color=INK_MUTED, fontsize=8)
    cb.outline.set_visible(False)

    # (B) weights -> runtime join.
    ax = axes[1]
    lay = np.repeat(np.arange(n_layers), nH)
    sc = ax.scatter(g_rms.ravel() * 1e3, openness.ravel(), c=lay, cmap=cmap,
                    s=7, alpha=0.7, linewidths=0)
    rho = sps.spearmanr(g_rms.ravel(), openness.ravel()).statistic
    within = np.median([sps.spearmanr(g_rms[L], openness[L]).statistic
                        for L in range(n_layers)])
    ax.text(0.60, 0.93, f"pooled ρ = {rho:+.2f}\nwithin-layer ρ = {within:+.2f}",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED)
    style_axes(ax, "Stronger gate weights → more-closed heads",
               "per-head gate_proj RMS (×10³, raw)", "mean gate openness")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("layer", color=INK_MUTED, fontsize=8)
    cb.outline.set_visible(False)

    fig.savefig(args.out_dir / "fig8_gate_openness.png", bbox_inches="tight")
    plt.close(fig)

    print(f"openness: min {openness.min():.3f} med {np.median(openness):.3f} "
          f"max {openness.max():.3f}")
    print(f"heads with frac_closed>0.5: {(closed > 0.5).sum()} "
          f"of {closed.size} ({(closed > 0.5).mean():.1%})")
    print(f"heads with openness<0.1: {(openness < 0.1).sum()} "
          f"({(openness < 0.1).mean():.1%})")


if __name__ == "__main__":
    main()

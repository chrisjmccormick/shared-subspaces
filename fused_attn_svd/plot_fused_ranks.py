"""Figures + printed stats for the fused-attention rank survey.

Reads results/<model>_sigmas.npz (written by fused_rank_survey.py) plus the
DS-R1 / Kimi-K2 singular-value pickles from the HF dataset
ChrisMcCormick/svd-attn-singvals, and produces:

  fig4_vo_rank_depth.png   the headline: per-head fused-VO effective rank vs
                           normalized depth for Glimmer / Qwen3-8B / DS-R1 / K2.
                           Median line + min-max band per layer.
  fig5_vo_sides.png        Glimmer per-side vs fused: V and O head ranks vs the
                           fused VO rank, by layer. Documents that neither Case-1
                           (bottleneck) nor Case-2 (misalignment) occurs.
  fig6_qk_nope.png         fused QK on the 13 NoPE layers (exact fusion), with the
                           Q/K per-side ranks. Scale-free qk_norm means any low
                           rank here would necessarily be Case-2 misalignment.
  fig7_gate_join.png       the norm-survey join: per-head gate RMS vs fused-VO
                           stable rank and vs fused-VO sigma_1.

Plus a stats block (also written to results/fused_rank_summary.txt) with the
numbers the write-up needs.

Usage:
    python plot_fused_ranks.py --results-dir results --out-dir .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy import stats as sps

import matplotlib.pyplot as plt

from plot_norms import GRID, INK_MUTED, SLOTS, label_ends, style_axes

# Colour follows the entity. Models keep one hue each (fig4); within-model
# matrix colours match plot_norms.py (q blue, k orange, v aqua, o yellow),
# and the FUSED map is always violet.
MODEL_COLOR = {"glimmer": SLOTS[0], "qwen3-8b": SLOTS[1],
               "ds-r1": SLOTS[2], "k2": SLOTS[3]}
MODEL_LABEL = {"glimmer": "Glimmer 30B", "qwen3-8b": "Qwen3-8B",
               "ds-r1": "DeepSeek-R1", "k2": "Kimi-K2"}
C_Q, C_K, C_V, C_O = SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3]
C_FUSED = SLOTS[6]

# Sequential blue ramp (light->dark) for depth encoding in scatter plots.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def eff_rank(S, err):
    e = np.asarray(S, dtype=np.float64) ** 2
    lost = 1.0 - np.cumsum(e) / e.sum()
    return int(np.argmax(lost <= err) + 1)


def eff_ranks(sig2d, err):
    """Per-row effective rank of a (heads, n_sig) array."""
    return np.array([eff_rank(s, err) for s in sig2d])


def stable_ranks(sig2d):
    e = np.asarray(sig2d, dtype=np.float64) ** 2
    return e.sum(axis=1) / e[:, 0]


def load_npz(results_dir: Path, model: str):
    z = dict(np.load(results_dir / f"{model}_sigmas.npz"))
    n_layers = json.loads(str(z["meta|config"]))["num_hidden_layers"]
    return z, n_layers


def load_pickle_vo(name: str, n_layers: int, key: str = "VO"):
    """Per-layer list of (heads, n_sig) arrays from the MLA-era pickles."""
    import pickle
    p = hf_hub_download("ChrisMcCormick/svd-attn-singvals",
                        f"{name}_singular_values.pkl", repo_type="dataset")
    with open(p, "rb") as f:
        _, S_heads, _ = pickle.load(f)
    return [np.stack(S_heads[L][key]) for L in range(n_layers)]


def band(ax, x, sig_by_layer, color, label, err=0.001):
    med, lo, hi = [], [], []
    for sig in sig_by_layer:
        r = eff_ranks(sig, err)
        med.append(np.median(r)); lo.append(r.min()); hi.append(r.max())
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, med, color=color, linewidth=2, label=label)
    return med


def fig4(models: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ends = []
    for name, sig_by_layer in models.items():
        n = len(sig_by_layer)
        x = np.arange(n) / (n - 1)
        med = band(ax, x, sig_by_layer, MODEL_COLOR[name], MODEL_LABEL[name])
        ends.append((med[-1], MODEL_LABEL[name], MODEL_COLOR[name]))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 135)
    ax.axhline(128, color=GRID, linewidth=0.8)
    ax.text(0.005, 129.5, "head dim = 128 (rank ceiling)",
            fontsize=7.5, color=INK_MUTED)
    style_axes(ax, "Fused value-output rank by depth: the DeepSeek/Kimi "
                   "low-rank pattern is absent in Glimmer",
               "layer / (n layers − 1)", "per-head fused VO effective rank "
                                         "(99.9% energy)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.legend(loc="lower center", frameon=False, ncol=4, fontsize=8)
    label_ends(ax, ends)
    fig.savefig(out / "fig4_vo_rank_depth.png", bbox_inches="tight")
    plt.close(fig)


def fig5(z: dict, n_layers: int, out: Path) -> None:
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    vo = [z[f"L{L}|sig_head|VO"] for L in x]
    band(ax, x, vo, C_FUSED, "fused VO (median + min–max)")
    o_med = [np.median(eff_ranks(z[f"L{L}|sig_head|O"], 0.001)) for L in x]
    ax.plot(x, o_med, color=C_O, linewidth=1.6, label="O heads (median)")
    v = np.array([eff_ranks(z[f"L{L}|sig_head|V"], 0.001) for L in x])
    for g, mk in ((0, "o"), (1, "s")):
        ax.scatter(x, v[:, g], s=9, marker=mk, color=C_V, alpha=0.8,
                   label=f"V head, KV group {g}")
    ax.set_ylim(0, 135)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.axhline(128, color=GRID, linewidth=0.8)
    style_axes(ax, "Glimmer: both sides and the fusion are near-full rank at "
                   "every layer — no bottleneck (Case 1), no misalignment (Case 2)",
               "layer", "effective rank (99.9% energy)")
    ax.legend(loc="lower center", frameon=False, ncol=4, fontsize=8)
    fig.savefig(out / "fig5_vo_sides.png", bbox_inches="tight")
    plt.close(fig)


def fig6(z: dict, nope: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xs = np.arange(len(nope))
    for j, L in enumerate(nope):
        qk = eff_ranks(z[f"L{L}|sig_head|QK"], 0.001)
        ax.scatter(np.full(len(qk), xs[j]) +
                   np.linspace(-0.18, 0.18, len(qk)),
                   qk, s=8, color=C_FUSED, alpha=0.65,
                   label="fused QK head" if j == 0 else None)
    q_med = [np.median(eff_ranks(z[f"L{L}|sig_head|Q"], 0.001)) for L in nope]
    k_min = [eff_ranks(z[f"L{L}|sig_head|K"], 0.001).min() for L in nope]
    ax.plot(xs, q_med, color=C_Q, linewidth=1.6, label="Q heads (median)")
    ax.plot(xs, k_min, color=C_K, linewidth=1.6, label="K heads (min of 2)")
    ax.set_xticks(xs, [str(L) for L in nope])
    ax.set_ylim(0, 135)
    ax.axhline(128, color=GRID, linewidth=0.8)
    style_axes(ax, "Glimmer NoPE layers: exact full-head QK fusion stays "
                   "near-full rank — no Case-2 misalignment either",
               "layer (the 13 NoPE / global-attention layers)",
               "effective rank (99.9% energy)")
    ax.legend(loc="lower center", frameon=False, ncol=3, fontsize=8)
    fig.savefig(out / "fig6_qk_nope.png", bbox_inches="tight")
    plt.close(fig)


def fig7(z: dict, n_layers: int, out: Path) -> None:
    g_rms, vo_sr, vo_s1, layer_of = [], [], [], []
    for L in range(n_layers):
        sig = z[f"L{L}|sig_head|VO"]
        g_rms.append(z[f"L{L}|rms_head|gate_proj"])
        vo_sr.append(stable_ranks(sig))
        vo_s1.append(sig[:, 0])
        layer_of.append(np.full(sig.shape[0], L))
    g_rms, vo_sr = np.concatenate(g_rms), np.concatenate(vo_sr)
    vo_s1, layer_of = np.concatenate(vo_s1), np.concatenate(layer_of)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9),
                             gridspec_kw={"wspace": 0.3})
    for ax, y, ylab, title in (
        (axes[0], vo_sr, "fused VO stable rank",
         "Stronger gates: more concentrated VO spectra"),
        (axes[1], vo_s1, "fused VO $\\sigma_1$",
         "…and larger VO amplitude"),
    ):
        sc = ax.scatter(g_rms * 1e3, y, c=layer_of, cmap=cmap, s=7, alpha=0.7,
                        linewidths=0)
        rho = sps.spearmanr(g_rms, y).statistic
        ax.text(0.02, 0.04, f"Spearman ρ = {rho:+.2f}", transform=ax.transAxes,
                fontsize=8, color=INK_MUTED)
        style_axes(ax, title, "per-head gate_proj RMS (×10³, raw)", ylab)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("layer", color=INK_MUTED)
    cb.outline.set_visible(False)
    fig.savefig(out / "fig7_gate_join.png", bbox_inches="tight")
    plt.close(fig)


def summarize(models: dict, z_g: dict, z_q: dict, nope: list[int],
              n_g: int, n_q: int) -> str:
    lines: list[str] = []
    say = lines.append

    say("== fused VO effective rank (99.9% energy), per-head ==")
    for name, sig_by_layer in models.items():
        n = len(sig_by_layer)
        early = slice(0, max(2, round(0.25 * n)))
        r_all = np.concatenate([eff_ranks(s, 0.001) for s in sig_by_layer])
        r_early = np.concatenate([eff_ranks(s, 0.001)
                                  for s in sig_by_layer[early]])
        say(f"{MODEL_LABEL[name]:>12}: overall min/med {r_all.min()}/"
            f"{int(np.median(r_all))} | first-25%-depth min/med "
            f"{r_early.min()}/{int(np.median(r_early))} | "
            f"heads<100: {(r_all < 100).mean():.1%} "
            f"(early {(r_early < 100).mean():.1%})")

    say("\n== Glimmer QK (per-head, 99.9% energy) ==")
    ex = np.concatenate([eff_ranks(z_g[f"L{L}|sig_head|QK"], 0.001)
                         for L in nope])
    sl = np.concatenate([eff_ranks(z_g[f"L{L}|sig_head|QK"], 0.001)
                         for L in range(n_g) if L not in nope])
    say(f"  NoPE layers (exact):     min/med/max {ex.min()}/"
        f"{int(np.median(ex))}/{ex.max()}  heads<100: {(ex < 100).mean():.1%}")
    say(f"  sliding (Δ=0 analysis):  min/med/max {sl.min()}/"
        f"{int(np.median(sl))}/{sl.max()}")

    say("\n== gate-norm join (Glimmer) ==")
    g_rms = np.stack([z_g[f"L{L}|rms_head|gate_proj"] for L in range(n_g)])
    vo = [z_g[f"L{L}|sig_head|VO"] for L in range(n_g)]
    er = np.stack([eff_ranks(s, 0.001) for s in vo])
    sr = np.stack([stable_ranks(s) for s in vo])
    s1 = np.stack([s[:, 0] for s in vo])
    o_rms = np.stack([z_g[f"L{L}|rms_head|o_proj"] for L in range(n_g)])
    for label, y in (("er999", er), ("stable rank", sr), ("sigma1", s1)):
        rho_all = sps.spearmanr(g_rms.ravel(), y.ravel()).statistic
        within = np.median([sps.spearmanr(g_rms[L], y[L]).statistic
                            for L in range(n_g)])
        say(f"  gate RMS vs VO {label:<12}: pooled ρ {rho_all:+.2f}, "
            f"median within-layer ρ {within:+.2f}")
    say(f"  gate RMS vs o_proj head RMS : pooled ρ "
        f"{sps.spearmanr(g_rms.ravel(), o_rms.ravel()).statistic:+.2f}, "
        f"median within-layer ρ "
        f"{np.median([sps.spearmanr(g_rms[L], o_rms[L]).statistic for L in range(n_g)]):+.2f}")

    say("\n== GQA group structure (Glimmer fused VO er999) ==")
    d_med = [abs(np.median(er[L, :16]) - np.median(er[L, 16:]))
             for L in range(n_g)]
    say(f"  |median(group0) − median(group1)| per layer: "
        f"max {max(d_med):.1f}, median {np.median(d_med):.1f}")

    say("\n== late-layer KV side (Glimmer) ==")
    for L in range(40, n_g):
        v = eff_ranks(z_g[f"L{L}|sig_head|V"], 0.001)
        k = eff_ranks(z_g[f"L{L}|sig_head|K"], 0.001)
        vr = z_g[f"L{L}|rms_head|v_proj"]
        say(f"  L{L}: V er999 {v.tolist()} K er999 {k.tolist()} "
            f"V head RMS ratio {vr.max()/vr.min():.2f}")

    say("\n== o_proj channel write amplitude (Glimmer) ==")
    frac_dead = []
    for L in range(n_g):
        cn = z_g[f"L{L}|colnorm|o_proj"]
        frac_dead.append((cn < 0.25 * np.median(cn)).mean())
    say(f"  channels with col-norm < 0.25×median: max over layers "
        f"{max(frac_dead):.2%} (layer {int(np.argmax(frac_dead))}), "
        f"model-wide mean {np.mean(frac_dead):.2%}")

    say("\n== stacked / whole-layer objects (Glimmer, er999) ==")
    for obj in ("VO_read", "VO_write", "QK_query", "QK_key"):
        key = f"sig_stacked|{obj}"
        vals = [eff_rank(z_g[f"L{L}|{key}"], 0.001)
                for L in range(n_g) if f"L{L}|{key}" in z_g]
        say(f"  {obj:<9}: min/med/max {min(vals)}/{int(np.median(vals))}/"
            f"{max(vals)} of {len(z_g[f'L0|{key}'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    z_g, n_g = load_npz(args.results_dir, "glimmer")
    z_q, n_q = load_npz(args.results_dir, "qwen3-8b")
    nope = z_g["meta|nope_layers"].tolist()

    models = {
        "glimmer": [z_g[f"L{L}|sig_head|VO"] for L in range(n_g)],
        "qwen3-8b": [z_q[f"L{L}|sig_head|VO"] for L in range(n_q)],
        "ds-r1": load_pickle_vo("ds-r1", 61),
        "k2": load_pickle_vo("k2", 61),
    }

    fig4(models, args.out_dir)
    fig5(z_g, n_g, args.out_dir)
    fig6(z_g, nope, args.out_dir)
    fig7(z_g, n_g, args.out_dir)

    text = summarize(models, z_g, z_q, nope, n_g, n_q)
    print(text)
    (args.results_dir / "fused_rank_summary.txt").write_text(text + "\n",
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()

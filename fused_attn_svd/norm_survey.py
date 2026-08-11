"""Per-matrix and per-head weight-RMS survey, streamed over the network (zero disk).

Answers two questions from the handoff doc:

  S5.3  At what granularity is the weight norm pinned? Kaiyue Wen's claim is that Muse
        Glimmer's per-matrix weight RMS is 0.5/sqrt(d_model), depending only on d and NOT
        on the individual matrix's fan-in/fan-out. Three hypotheses are separated by
        comparing matrices of very different shapes (q_proj 4096x6656 vs mlp.down_proj
        6656x19968) and by checking whether per-ROW norms are constant:
          - Hyperball (fixed ||W||_F per matrix): one constant across all shapes;
            rows free to vary.
          - nGPT-style (rows/cols normalized): per-row norms near-constant.
          - Spectral condition (plain Muon): RMS varies with shape.

  S5.1  Is the statistic true at a HEAD level? Heads partition the rows of q/k/v/gate_proj
        and the columns of o_proj exactly, so RMS(W)^2 is precisely the mean of the
        per-head RMS^2 -- a pinned whole-matrix RMS is structurally blind to per-head
        spread. This reports that spread directly.

Usage:
    python norm_survey.py                          # Glimmer, a sample of layers
    python norm_survey.py --layers all             # every layer (~50 GB of ranged reads)
    python norm_survey.py --model qwen3-8b         # the contrast model
    python norm_survey.py --csv out.csv            # tidy per-matrix rows
    python norm_survey.py --npz out.npz            # full per-head RMS + per-row norms

The .npz is worth writing on any long run: a full sweep is ~50 GB of network reads, and
the per-head/per-row arrays are what the plots need. Keys are "<layer>|<matrix>|head" and
"<layer>|<matrix>|row".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
from huggingface_hub import HfFileSystem

from hf_stream import RemoteSafetensors

MODELS = {
    "glimmer": {
        "repo": "meta-models/Muse-Glimmer-30B",
        "prefix": "model.language_model.layers",
        "cfg_key": "text_config",
        # name -> (head axis: "rows"|"cols"|None)
        "matrices": {
            "self_attn.q_proj": "rows",
            "self_attn.k_proj": "rows",
            "self_attn.v_proj": "rows",
            "self_attn.gate_proj": "rows",
            "self_attn.o_proj": "cols",
            "mlp.gate_proj": None,
            "mlp.up_proj": None,
            "mlp.down_proj": None,
        },
    },
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B",
        "prefix": "model.layers",
        "cfg_key": None,
        "matrices": {
            "self_attn.q_proj": "rows",
            "self_attn.k_proj": "rows",
            "self_attn.v_proj": "rows",
            "self_attn.o_proj": "cols",
            "mlp.gate_proj": None,
            "mlp.up_proj": None,
            "mlp.down_proj": None,
        },
    },
}


def head_rms(W: torch.Tensor, axis: str | None, head_dim: int) -> torch.Tensor | None:
    """Per-head RMS. Heads are contiguous blocks of rows (q/k/v/gate) or columns (o)."""
    if axis is None:
        return None
    M = W if axis == "rows" else W.T          # (n_heads*head_dim, other)
    n_heads = M.shape[0] // head_dim
    if n_heads < 2:
        return None
    return M.reshape(n_heads, head_dim, -1).pow(2).mean(dim=(1, 2)).sqrt()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glimmer", choices=sorted(MODELS))
    ap.add_argument("--layers", default="sample",
                    help="'sample' (default), 'all', or a comma list like 0,3,25,51")
    ap.add_argument("--csv", default=None, help="write tidy per-matrix rows here")
    ap.add_argument("--npz", default=None,
                    help="write full per-head RMS and per-row norm arrays here")
    args = ap.parse_args()

    spec = MODELS[args.model]
    fs = HfFileSystem()
    cfg = json.loads(fs.read_text(f"{spec['repo']}/config.json"))
    if spec["cfg_key"]:
        cfg = cfg[spec["cfg_key"]]

    d = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim", d // cfg["num_attention_heads"])
    target = 0.5 / d**0.5

    if args.layers == "all":
        layers = list(range(n_layers))
    elif args.layers == "sample":
        layers = sorted({0, 1, 2, 3, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 1})
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"model      : {args.model}  ({spec['repo']})")
    print(f"d_model    : {d}   layers: {n_layers}   head_dim: {head_dim}")
    print(f"q/kv heads : {cfg['num_attention_heads']} / {cfg['num_key_value_heads']}")
    print(f"0.5/sqrt(d): {target:.6e}")
    print(f"layers     : {layers}\n")

    hdr = (f"{'layer':>5} {'matrix':<20}{'shape':>16}{'RMS':>12}{'/target':>9}"
           f"{'head RMS min':>14}{'max':>12}{'max/min':>9}{'row CV':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    seen_bytes = 0

    # Flush after every layer, not at the end: a full sweep is ~50 GB of network reads
    # and the box it runs on may be preemptible. Losing 40 minutes to a dropped pod
    # because the results only landed on exit would be a self-inflicted wound.
    def flush() -> None:
        if args.csv and rows:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        if args.npz and arrays:
            np.savez_compressed(args.npz, **arrays)

    ck = RemoteSafetensors(spec["repo"])
    try:
        for L in layers:
            for mat, axis in spec["matrices"].items():
                name = f"{spec['prefix']}.{L}.{mat}.weight"
                if name not in ck.weight_map:
                    continue
                seen_bytes += ck.nbytes(name)
                W = ck.get(name, dtype=torch.float32)

                rms = W.pow(2).mean().sqrt().item()
                # per-row norms: the nGPT discriminator
                row_norms = W.pow(2).sum(dim=1).sqrt()
                row_cv = (row_norms.std() / row_norms.mean()).item()

                h = head_rms(W, axis, head_dim)
                if h is None:
                    hmin = hmax = ratio = float("nan")
                else:
                    hmin, hmax = h.min().item(), h.max().item()
                    ratio = hmax / hmin
                    if args.npz:
                        arrays[f"{L}|{mat}|head"] = h.numpy()
                if args.npz:
                    arrays[f"{L}|{mat}|row"] = row_norms.numpy().astype(np.float32)

                print(f"{L:>5} {mat:<20}{str(tuple(W.shape)):>16}{rms:>12.4e}"
                      f"{rms / target:>9.4f}{hmin:>14.4e}{hmax:>12.4e}"
                      f"{ratio:>9.2f}{row_cv:>9.4f}")
                rows.append(dict(layer=L, matrix=mat, shape=tuple(W.shape), rms=rms,
                                 ratio_to_target=rms / target, head_rms_min=hmin,
                                 head_rms_max=hmax, head_max_over_min=ratio,
                                 row_norm_cv=row_cv))
                del W, row_norms
            el = time.time() - t0
            gb = seen_bytes / 2**30
            done = layers.index(L) + 1
            eta = el / done * (len(layers) - done)
            print(f"      [layer {done}/{len(layers)}  {gb:.1f} GiB read  "
                  f"{el/60:.1f} min elapsed  ~{eta/60:.1f} min left]", flush=True)
            print()
            flush()
    finally:
        ck.close()
        flush()

    if args.csv:
        print(f"wrote {len(rows)} rows -> {args.csv}")
    if args.npz:
        print(f"wrote {len(arrays)} arrays -> {args.npz}")


if __name__ == "__main__":
    main()

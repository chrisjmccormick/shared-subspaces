"""Per-head fused-attention rank survey, streamed over the network (zero disk).

Computes, for every attention layer of the target model:

  per-head singular values   Q_i, K_g, V_g, O_i, G_i (gate, if present)
  fused VO                   W^VO_i = W^V_{g(i)} W^O_i        -- all layers
  fused QK                   W^QK_i = W^Q_i (W^K_{g(i)})^T    -- exact on NoPE layers
  whole-matrix spectra       q/k/v/o/gate_proj
  stacked-head spectra       per-side (heads /sigma_1, concatenated) and both Gram
                             views of the fused maps (read side and write side)
  raw per-head weight RMS    the join key against the norm-survey results
  o_proj column norms        per head-channel write amplitude into the residual

Nothing 6656x6656 is ever materialized. For a fused product A B^T with
A, B in R^{d x r}: thin-QR both sides, A = Q_A R_A, B = Q_B R_B, then
svdvals(A B^T) = svdvals(R_A R_B^T) -- an r x r problem (r = 128 here).
The stacked/Gram spectra reduce the same way (see comments at the call sites).

Norm-fold gotcha (handoff S3.5): Glimmer's RMSNorm gamma is ZERO-CENTERED,
so the fold into the input-consuming matrices is W * (1 + gamma), not W * gamma.
Qwen3 uses standard gamma. Per-head RMS is reported on the RAW weights (matching
norm_survey.py); spectra are computed on the folded weights (the functional map
from the RMS-normalized residual stream).

Usage:
    python fused_rank_survey.py --model glimmer --layers all
    python fused_rank_survey.py --model qwen3-8b --layers 0,3
Outputs (flushed after every layer; the box may be preemptible):
    results/<model>_fused_ranks_heads.csv   tidy per-head effective ranks
    results/<model>_fused_ranks_layer.csv   whole-matrix + stacked effective ranks
    results/<model>_sigmas.npz              full singular-value arrays
"""

from __future__ import annotations

import argparse
import csv
import json
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
        "gate": True,
        "centered_gamma": True,   # fold = 1 + gamma
    },
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B",
        "prefix": "model.layers",
        "cfg_key": None,
        "gate": False,
        "centered_gamma": False,  # fold = gamma
    },
}


def get_rank_for_error_threshold(S_values, error_threshold):
    """Minimum rank keeping energy loss below `error_threshold` (notebook-canonical).

    Energy in float64: squaring fp32 sigmas overflows/loses tail precision.
    """
    energy = np.asarray(S_values, dtype=np.float64) ** 2
    fraction_lost = 1.0 - np.cumsum(energy) / energy.sum()
    return int(np.argmax(fraction_lost <= error_threshold) + 1)


def stable_rank(S_values) -> float:
    e = np.asarray(S_values, dtype=np.float64) ** 2
    return float(e.sum() / e[0])


def psd_sqrt_factor(M: torch.Tensor) -> torch.Tensor:
    """L with L L^T = M for symmetric PSD M (eigh; safe where Cholesky isn't)."""
    w, E = torch.linalg.eigh(M)
    return E * w.clamp_min(0).sqrt().unsqueeze(-2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glimmer", choices=sorted(MODELS))
    ap.add_argument("--layers", default="all",
                    help="'all' (default) or a comma list like 0,3,25,51")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    spec = MODELS[args.model]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # These are SVD inputs -- keep matmuls in real fp32, not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    fs = HfFileSystem()
    cfg = json.loads(fs.read_text(f"{spec['repo']}/config.json"))
    if spec["cfg_key"]:
        cfg = cfg[spec["cfg_key"]]

    d = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    nH = cfg["num_attention_heads"]
    nKV = cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", d // nH)
    group = nH // nKV
    # HF repeat_kv repeats each KV head contiguously: query head i -> kv head i//group.
    idx_g = torch.arange(nH, device=dev) // group
    rope_theta = cfg.get("layer_rope_theta")
    nope_layers = ([i for i, t in enumerate(rope_theta) if not t]
                   if rope_theta else [])

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"model : {args.model} ({spec['repo']})  device: {dev}")
    print(f"d={d} layers={n_layers} heads={nH}/{nKV} head_dim={hd} group={group}")
    print(f"NoPE layers (exact QK fusion): {nope_layers or 'none -- QK skipped'}")
    print(f"layers: {layers}\n")

    head_rows: list[dict] = []
    layer_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {
        "meta|nope_layers": np.array(nope_layers, dtype=np.int64),
        "meta|config": np.array(json.dumps(
            {k: cfg[k] for k in ("hidden_size", "num_hidden_layers",
                                 "num_attention_heads", "num_key_value_heads")}
            | {"head_dim": hd, "model": args.model})),
    }
    heads_csv = f"{args.outdir}/{args.model}_fused_ranks_heads.csv"
    layer_csv = f"{args.outdir}/{args.model}_fused_ranks_layer.csv"
    npz_path = f"{args.outdir}/{args.model}_sigmas.npz"

    def flush() -> None:
        for path, rows in ((heads_csv, head_rows), (layer_csv, layer_rows)):
            if rows:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        np.savez_compressed(npz_path, **arrays)

    def head_row(L: int, mat: str, i: int, g: int, sig: np.ndarray,
                 rms: float | None = None, exact: bool | None = None) -> dict:
        return dict(layer=L, matrix=mat, head=i, kv_group=g,
                    sigma1=float(sig[0]), stable_rank=round(stable_rank(sig), 3),
                    er99=get_rank_for_error_threshold(sig, 0.01),
                    er999=get_rank_for_error_threshold(sig, 0.001),
                    head_rms_raw="" if rms is None else f"{rms:.6e}",
                    exact="" if exact is None else int(exact))

    def layer_row(L: int, obj: str, sig: np.ndarray) -> dict:
        return dict(layer=L, object=obj, n_sigmas=len(sig), sigma1=float(sig[0]),
                    er99=get_rank_for_error_threshold(sig, 0.01),
                    er999=get_rank_for_error_threshold(sig, 0.001))

    def save_sig(L: int, kind: str, name: str, sig: torch.Tensor) -> np.ndarray:
        a = sig.cpu().numpy().astype(np.float32)
        arrays[f"L{L}|{kind}|{name}"] = a
        return a

    ck = RemoteSafetensors(spec["repo"])
    t0 = time.time()
    seen_bytes = 0
    try:
        for L in layers:
            pre = f"{spec['prefix']}.{L}"
            W = {}
            mats = ["q_proj", "k_proj", "v_proj", "o_proj"] + (
                ["gate_proj"] if spec["gate"] else [])
            for m in mats:
                name = f"{pre}.self_attn.{m}.weight"
                seen_bytes += ck.nbytes(name)
                W[m] = ck.get(name, dtype=torch.float32).to(dev)
            gamma = ck.get(f"{pre}.input_layernorm.weight",
                           dtype=torch.float32).to(dev)
            fold = (1.0 + gamma) if spec["centered_gamma"] else gamma

            if L == layers[0]:
                assert W["q_proj"].shape == (nH * hd, d), W["q_proj"].shape
                assert W["k_proj"].shape == (nKV * hd, d), W["k_proj"].shape
                assert W["o_proj"].shape == (d, nH * hd), W["o_proj"].shape
                rms0 = W["q_proj"].pow(2).mean().sqrt().item()
                print(f"[assert PASS] shapes ok; layer-{L} q_proj RMS {rms0:.4e} "
                      f"(x sqrt(d)/0.5 = {rms0 * d**0.5 / 0.5:.4f})")

            # -- raw norms (pre-fold), matching norm_survey.py conventions ------
            for m, axis in (("q_proj", 0), ("k_proj", 0), ("v_proj", 0),
                            ("gate_proj", 0), ("o_proj", 1)):
                if m not in W:
                    continue
                M = W[m] if axis == 0 else W[m].T
                r = M.reshape(-1, hd, M.shape[-1]).pow(2).mean(dim=(1, 2)).sqrt()
                arrays[f"L{L}|rms_head|{m}"] = r.cpu().numpy().astype(np.float32)
            arrays[f"L{L}|colnorm|o_proj"] = (
                W["o_proj"].pow(2).sum(dim=0).sqrt().cpu().numpy().astype(np.float32))
            arrays[f"L{L}|fold"] = fold.cpu().numpy().astype(np.float32)

            # -- fold the input norm into everything that reads the residual ----
            for m in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                if m in W:
                    W[m] = W[m] * fold  # (out, in) * (in,)

            # -- per-head tall matrices (d, hd), xW convention ------------------
            A = {"Q": W["q_proj"].view(nH, hd, d).transpose(1, 2),
                 "K": W["k_proj"].view(nKV, hd, d).transpose(1, 2),
                 "V": W["v_proj"].view(nKV, hd, d).transpose(1, 2),
                 "O": W["o_proj"].view(d, nH, hd).permute(1, 0, 2)}
            if spec["gate"]:
                A["G"] = W["gate_proj"].view(nH, hd, d).transpose(1, 2)

            # Thin QR per side; per-head sigmas come free as svdvals(R).
            R = {k: torch.linalg.qr(v, mode="reduced")[1] for k, v in A.items()}
            sig_head = {k: torch.linalg.svdvals(r) for k, r in R.items()}

            # -- fused products via R-products ---------------------------------
            sig_vo = torch.linalg.svdvals(R["V"][idx_g] @ R["O"].transpose(1, 2))
            do_qk = bool(nope_layers)  # model has NoPE layers at all
            if do_qk:
                sig_qk = torch.linalg.svdvals(R["Q"] @ R["K"][idx_g].transpose(1, 2))

            # -- record per-head results ---------------------------------------
            mat_of = {"Q": "q_proj", "K": "k_proj", "V": "v_proj",
                      "O": "o_proj", "G": "gate_proj"}
            for k, sig in sig_head.items():
                a = save_sig(L, "sig_head", k, sig)
                rms = arrays[f"L{L}|rms_head|{mat_of[k]}"]
                nh_k = a.shape[0]
                for i in range(nh_k):
                    g = i if nh_k == nKV else int(i // group)
                    head_rows.append(head_row(L, k, i, g, a[i], rms=float(rms[i])))
            a = save_sig(L, "sig_head", "VO", sig_vo)
            for i in range(nH):
                head_rows.append(head_row(L, "VO", i, int(i // group), a[i]))
            if do_qk:
                a = save_sig(L, "sig_head", "QK", sig_qk)
                for i in range(nH):
                    head_rows.append(head_row(L, "QK", i, int(i // group), a[i],
                                              exact=L in nope_layers))

            # -- whole-matrix spectra ------------------------------------------
            for m in mats:
                sig = torch.linalg.svdvals(W[m])
                save_sig(L, "sig_full", m, sig)
                layer_rows.append(layer_row(L, f"{m}_full", sig.cpu().numpy()))

            # -- stacked heads (each / its sigma_1, concatenated) ---------------
            for k in A:
                s1 = sig_head[k][:, :1, None]
                stk = (A[k] / s1).permute(1, 0, 2).reshape(d, -1)
                sig = torch.linalg.svdvals(stk)
                save_sig(L, "sig_stacked", k, sig)
                layer_rows.append(layer_row(L, f"{k}_stacked", sig.cpu().numpy()))

            # Fused-map Gram views, factored so nothing dxd appears:
            #   write side: sum_i Wvo_i^T Wvo_i = D^T D,  D_i = R_V[g] B_i / s1_i
            #   read side:  sum_i Wvo_i Wvo_i^T = sum_g A_V[g] N_g A_V[g]^T,
            #               N_g = sum_{i in g} R_O[i]^T R_O[i] / s1_i^2
            s1 = sig_vo[:, :1, None]
            Dm = (R["V"][idx_g] @ A["O"].transpose(1, 2)) / s1
            sig = torch.linalg.svdvals(Dm.reshape(nH * hd, d))
            save_sig(L, "sig_stacked", "VO_write", sig)
            layer_rows.append(layer_row(L, "VO_stacked_write", sig.cpu().numpy()))

            N = (R["O"].transpose(1, 2) @ R["O"]) / (s1 * s1)
            N = N.view(nKV, group, hd, hd).sum(dim=1)
            Fm = torch.cat([A["V"][g] @ psd_sqrt_factor(N[g]) for g in range(nKV)],
                           dim=1)
            sig = torch.linalg.svdvals(Fm)
            save_sig(L, "sig_stacked", "VO_read", sig)
            layer_rows.append(layer_row(L, "VO_stacked_read", sig.cpu().numpy()))

            if do_qk:
                s1 = sig_qk[:, :1, None]
                Fq = ((A["Q"] @ R["K"][idx_g].transpose(1, 2)) / s1)
                sig = torch.linalg.svdvals(Fq.permute(1, 0, 2).reshape(d, -1))
                save_sig(L, "sig_stacked", "QK_query", sig)
                layer_rows.append(layer_row(L, "QK_stacked_query", sig.cpu().numpy()))

                M = (R["Q"].transpose(1, 2) @ R["Q"]) / (s1 * s1)
                M = M.view(nKV, group, hd, hd).sum(dim=1)
                Fk = torch.cat([A["K"][g] @ psd_sqrt_factor(M[g])
                                for g in range(nKV)], dim=1)
                sig = torch.linalg.svdvals(Fk)
                save_sig(L, "sig_stacked", "QK_key", sig)
                layer_rows.append(layer_row(L, "QK_stacked_key", sig.cpu().numpy()))

            del W, A, R
            done = layers.index(L) + 1
            el, gb = time.time() - t0, seen_bytes / 2**30
            vo99 = [r["er999"] for r in head_rows
                    if r["layer"] == L and r["matrix"] == "VO"]
            print(f"L{L:>3}  VO er999 min/med/max "
                  f"{min(vo99):>3}/{int(np.median(vo99)):>3}/{max(vo99):>3}"
                  f"   [{done}/{len(layers)}  {gb:.1f} GiB  {el/60:.1f} min"
                  f"  ~{el/done*(len(layers)-done)/60:.1f} min left]", flush=True)
            flush()
    finally:
        ck.close()
        flush()

    print(f"\nwrote {len(head_rows)} head rows -> {heads_csv}")
    print(f"wrote {len(layer_rows)} layer rows -> {layer_csv}")
    print(f"wrote {len(arrays)} arrays -> {npz_path}")


if __name__ == "__main__":
    main()

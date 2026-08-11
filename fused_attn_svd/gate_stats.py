"""Per-head attention-gate statistics for Muse Glimmer 30B on calibration text.

The weights-only survey (fused_rank_survey.py) shows Glimmer has none of the
low-rank head suppression seen in DeepSeek/Kimi. The architecture's remaining
suppression channel is the data-dependent output gate
`attn_out * sigmoid(gate_proj(x))`. This measures what that gate actually does
on real text, per head:

  openness    E_t[mean_c sigmoid(g)]      1.0 = fully open, 0.0 = head off
  switchiness std_t of per-token openness  large = the head toggles per token
  frac_closed fraction of tokens with openness < 0.1  (head turned off)
  ch_lo/ch_hi per-channel saturation fractions (sigmoid < 0.1 / > 0.9)

Per-channel mean gates are saved too: a channel whose mean gate is ~0 is a
runtime-suppressed inner direction -- the data-dependent analogue of the small
singular values that DeepSeek expresses in its weights.

Calibration: wikitext-103 train text, chunked to fixed-length sequences.
Runs the full text decoder (the vision tower is loaded but unused).

Usage:
    python gate_stats.py --n-seqs 128 --seq-len 2048
Outputs:
    results/glimmer_gate_stats.npz
    results/glimmer_gate_stats.csv   per layer x head summary
"""

from __future__ import annotations

import argparse
import csv
import re
import time

import numpy as np
import torch


def calibration_ids(tokenizer, n_seqs: int, seq_len: int) -> torch.Tensor:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Salesforce/wikitext",
                        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                        repo_type="dataset")
    text = "\n\n".join(t for t in pd.read_parquet(p)["text"] if t.strip())
    need = n_seqs * seq_len
    # ~4 chars/token with plenty of slack; tokenizing the whole shard is waste.
    ids = tokenizer(text[: need * 8], return_tensors="pt").input_ids[0][:need]
    assert len(ids) == need, f"only {len(ids)} tokens from calibration shard"
    return ids.view(n_seqs, seq_len)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--n-seqs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    batches = calibration_ids(tokenizer, args.n_seqs, args.seq_len).split(args.batch)

    model = AutoModelForImageTextToText.from_pretrained(
        args.repo, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    gates = {}  # layer index -> gate_proj module
    for name, mod in model.named_modules():
        m = re.search(r"language_model\.layers\.(\d+)\.self_attn\.gate_proj$", name)
        if m:
            gates[int(m.group(1))] = mod
    n_layers = len(gates)
    hd = model.config.text_config.head_dim
    nH = model.config.text_config.num_attention_heads
    print(f"hooking {n_layers} gate_proj modules; {nH} heads x {hd} dims")

    acc = {L: dict(n=0,
                   ch_sum=torch.zeros(nH, hd, device="cuda"),
                   ch_lo=torch.zeros(nH, hd, device="cuda"),
                   ch_hi=torch.zeros(nH, hd, device="cuda"),
                   open_sum=torch.zeros(nH, device="cuda"),
                   open_sumsq=torch.zeros(nH, device="cuda"),
                   closed=torch.zeros(nH, device="cuda"))
           for L in gates}

    def hook(L):
        def fn(_mod, _inp, out):
            g = torch.sigmoid(out.float()).reshape(-1, nH, hd)
            a = acc[L]
            a["n"] += g.shape[0]
            a["ch_sum"] += g.sum(0)
            a["ch_lo"] += (g < 0.1).float().sum(0)
            a["ch_hi"] += (g > 0.9).float().sum(0)
            openness = g.mean(-1)
            a["open_sum"] += openness.sum(0)
            a["open_sumsq"] += openness.pow(2).sum(0)
            a["closed"] += (openness < 0.1).float().sum(0)
        return fn

    handles = [gates[L].register_forward_hook(hook(L)) for L in gates]
    t0 = time.time()
    with torch.no_grad():
        for bi, ids in enumerate(batches):
            model(input_ids=ids.to("cuda"), use_cache=False)
            if (bi + 1) % 8 == 0 or bi + 1 == len(batches):
                el = time.time() - t0
                print(f"  batch {bi+1}/{len(batches)}  {el/60:.1f} min", flush=True)
    for h in handles:
        h.remove()

    arrays: dict[str, np.ndarray] = {}
    rows = []
    for L, a in acc.items():
        n = a["n"]
        ch_mean = (a["ch_sum"] / n).cpu().numpy()
        ch_lo = (a["ch_lo"] / n).cpu().numpy()
        ch_hi = (a["ch_hi"] / n).cpu().numpy()
        op_mean = (a["open_sum"] / n).cpu().numpy()
        op_std = ((a["open_sumsq"] / n).cpu().numpy() - op_mean**2).clip(0) ** 0.5
        closed = (a["closed"] / n).cpu().numpy()
        arrays[f"L{L}|ch_mean"] = ch_mean.astype(np.float32)
        arrays[f"L{L}|ch_lo"] = ch_lo.astype(np.float32)
        arrays[f"L{L}|ch_hi"] = ch_hi.astype(np.float32)
        for i in range(nH):
            rows.append(dict(
                layer=L, head=i,
                openness=round(float(op_mean[i]), 4),
                switchiness=round(float(op_std[i]), 4),
                frac_closed=round(float(closed[i]), 4),
                ch_lo_frac=round(float(ch_lo[i].mean()), 4),
                ch_hi_frac=round(float(ch_hi[i].mean()), 4),
                min_ch_mean=round(float(ch_mean[i].min()), 4),
                n_ch_mean_lt_01=int((ch_mean[i] < 0.1).sum()),
            ))
    arrays["meta|n_tokens"] = np.array([next(iter(acc.values()))["n"]])

    np.savez_compressed(f"{args.outdir}/glimmer_gate_stats.npz", **arrays)
    with open(f"{args.outdir}/glimmer_gate_stats.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows over {arrays['meta|n_tokens'][0]} tokens")


if __name__ == "__main__":
    main()

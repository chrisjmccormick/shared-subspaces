"""Read individual tensors out of a remote safetensors checkpoint via HTTP range reads.

Motivation: the fused-attention analyses are *weights-only*, and for Muse Glimmer 30B the
attention tensors (q/k/v/o/gate_proj + the layernorms) are ~8.9 GB of a 59.6 GB checkpoint.
Downloading the other 51 GB of MLP and embedding weights to touch none of it is pure waste
-- and on a 50 GB box it is not even possible.

safetensors files begin with an 8-byte little-endian header length, then a JSON header
mapping tensor name -> {dtype, shape, data_offsets}, then the raw tensor bytes. Every
tensor is therefore one contiguous byte range, so a single ranged GET fetches exactly one
tensor. `huggingface_hub.HfFileSystem` gives a seekable file object that turns `seek`/`read`
into range requests against the CDN, so nothing is written to disk at all.

Two access patterns:

    ckpt = RemoteSafetensors("meta-models/Muse-Glimmer-30B")
    W = ckpt.get("model.language_model.layers.0.self_attn.q_proj.weight")   # one tensor

    for name, W in ckpt.iter_tensors(lambda n: n.endswith("q_proj.weight")):
        ...        # streamed one at a time; nothing accumulates

`iter_tensors` is the one to reach for when computing statistics over the *whole*
checkpoint (e.g. per-matrix weight RMS): it never holds more than one tensor at a time, so
peak disk is zero and peak memory is one matrix.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import torch
from huggingface_hub import HfFileSystem

# safetensors dtype strings -> torch dtypes
_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"


class RemoteSafetensors:
    """Random access into a (possibly sharded) safetensors checkpoint on the HF Hub.

    Shard headers are fetched once and cached; file handles are kept open and reused, so
    reading many tensors from one shard costs one connection, not one per tensor.
    """

    def __init__(self, repo_id: str, revision: str = "main", repo_type: str = "model"):
        self.repo_id = repo_id
        self.revision = revision
        self._fs = HfFileSystem()
        prefix = "" if repo_type == "model" else f"{repo_type}s/"
        self._root = f"{prefix}{repo_id}@{revision}" if revision != "main" else f"{prefix}{repo_id}"

        # weight_map: tensor name -> shard filename
        try:
            index = json.loads(self._fs.read_text(f"{self._root}/{_INDEX_NAME}"))
            self.weight_map: dict[str, str] = index["weight_map"]
        except FileNotFoundError:
            # Unsharded checkpoint: one file, every tensor in it.
            self.weight_map = {}
            self._single = True
        else:
            self._single = False

        self._headers: dict[str, dict] = {}   # shard -> parsed JSON header
        self._offsets: dict[str, int] = {}    # shard -> byte offset of the data section
        self._handles: dict[str, object] = {}

        if self._single:
            hdr = self._header(_SINGLE_NAME)
            self.weight_map = {k: _SINGLE_NAME for k in hdr if k != "__metadata__"}

    # -- internals ----------------------------------------------------------

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = self._fs.open(f"{self._root}/{shard}", "rb")
        return self._handles[shard]

    def _header(self, shard: str) -> dict:
        if shard not in self._headers:
            f = self._handle(shard)
            f.seek(0)
            n = int.from_bytes(f.read(8), "little")
            self._headers[shard] = json.loads(f.read(n))
            self._offsets[shard] = 8 + n
        return self._headers[shard]

    # -- public API ---------------------------------------------------------

    def keys(self) -> list[str]:
        """Every tensor name in the checkpoint, in weight-map order."""
        return list(self.weight_map)

    def info(self, name: str) -> dict:
        """`{'dtype': ..., 'shape': [...], 'data_offsets': [...]}` without fetching data."""
        shard = self.weight_map[name]
        return self._header(shard)[name]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.info(name)["shape"])

    def nbytes(self, name: str) -> int:
        start, end = self.info(name)["data_offsets"]
        return end - start

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Fetch one tensor. `dtype` casts after load (e.g. float32 for SVD work)."""
        shard = self.weight_map[name]
        entry = self._header(shard)[name]
        start, end = entry["data_offsets"]
        base = self._offsets[shard]

        f = self._handle(shard)
        f.seek(base + start)
        raw = f.read(end - start)
        if len(raw) != end - start:
            raise IOError(f"short read for {name}: got {len(raw)} of {end - start} bytes")

        # frombuffer needs a writable buffer to avoid a UserWarning; bytearray gives one
        # without a second copy of the data.
        t = torch.frombuffer(bytearray(raw), dtype=_DTYPES[entry["dtype"]])
        t = t.reshape(tuple(entry["shape"]))
        return t.to(dtype) if dtype is not None else t

    def iter_tensors(
        self,
        predicate: Callable[[str], bool] | None = None,
        dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream matching tensors one at a time, grouped by shard to reuse connections.

        Nothing accumulates -- use this for whole-checkpoint statistics on a small disk.
        """
        names = [n for n in self.weight_map if predicate is None or predicate(n)]
        names.sort(key=lambda n: (self.weight_map[n], self.info(n)["data_offsets"][0]))
        for name in names:
            yield name, self.get(name, dtype=dtype)

    def close(self) -> None:
        for f in self._handles.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        self._handles.clear()

    def __enter__(self) -> "RemoteSafetensors":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

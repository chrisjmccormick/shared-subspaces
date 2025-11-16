"""
# Sparse MoE Package

This package mirrors the structure of `subspace_decoder` but swaps the dense
feed-forward blocks for a sparse mixture-of-experts implementation. Modules
are grouped under `configs/`, `layers/`, and `models/` to keep parity with
the original codebase while making it easy to experiment with MoE-specific
components in isolation.
"""

from . import configs  # noqa: F401
from . import layers   # noqa: F401
from . import models   # noqa: F401

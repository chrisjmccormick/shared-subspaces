"""
# Models

Expose decoder architectures that integrate Sparse MoE blocks. The API mirrors
`subspace_decoder.models` so model creation flows remain identical while we
experiment with expert routing.
"""

from .shared_space_config import SparseMoEDecoderConfig  # noqa: F401
from .shared_space_decoder import SparseMoEDecoder  # noqa: F401

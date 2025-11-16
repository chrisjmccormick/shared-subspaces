"""
# Layers

Re-export the key layer building blocks (MLA attention, Sparse MoE feed-forward
stack, and shared utilities) so downstream code can import from
`moe.layers` without reaching into submodules directly.
"""

from .mla import *  # noqa: F401,F403
from .feedforward import *  # noqa: F401,F403
from .task_heads import *  # noqa: F401,F403

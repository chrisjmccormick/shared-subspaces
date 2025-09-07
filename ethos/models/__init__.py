"""
ETHOS Models

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_ethos": ["EthosConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ethos"] = [
        "EthosPreTrainedModel",
        "EthosModel", 
        "EthosForCausalLM",
    ]

if TYPE_CHECKING:
    from .configuration_ethos import EthosConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ethos import (
            EthosForCausalLM,
            EthosModel,
            EthosPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

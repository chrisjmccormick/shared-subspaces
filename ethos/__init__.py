"""
ETHOS: A Novel Mixture-of-Experts Architecture

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

from .models import EthosConfig, EthosModel, EthosForCausalLM, EthosPreTrainedModel

__version__ = "0.1.0"

__all__ = [
    "EthosConfig",
    "EthosModel", 
    "EthosForCausalLM",
    "EthosPreTrainedModel",
]

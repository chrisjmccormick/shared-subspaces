
# ETHOS: Efficient Transformers via Hypernetwork-Organized Sparsity

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository contains the implementation of ETHOS from the paper "ETHOS: Efficient Transformers via
Hypernetwork-Organized Sparsity"

ETHOS is a novel architecture that dynamically generates millions of tiny experts from compressed latent representations, achieving 8.7B parameter capacity while using ~20× fewer FLOPs.

### Model Architecture

ETHOS combines several key innovations:
- **Dynamic expert generation**: Instead of storing millions of expert parameters, we generate them from 128-dimensional latent codes
- **Product-key routing**: Efficient O(√N) routing to 262K experts per layer utilizing Query BatchNorm from PEER
- **Reordered execution**: Custom Triton kernel achieving 8× speedup
- **GPT-2 Tokenizer**: Uses the standard GPT-2 tokenizer (vocab size: 50,257) for text processing

### Repository Structure

TODO

### Configuration

Key parameters:
- `num_experts`: 262,144 (512²) experts per layer
- `d_latent`: 128-dimensional latent codes
- `top_k`: 16 experts selected per token
- `num_routing_heads`: 8 independent routing heads


### Requirements

- PyTorch 2.0+
- CUDA 11.8+
- Triton 2.1+
- Flash Attention 2.0+
- 80GB+ GPU memory recommended


## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** - see the [LICENSE](LICENSE) file for details.

**Important**: The AGPLv3 license requires that any modifications or derivative works be released under the same license, including when used as a network service.

### Commercial Licensing

For commercial use cases that require a different license, please contact **wryanmedford@gmail.com** to discuss commercial licensing options.


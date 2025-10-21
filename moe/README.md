## Model Architecture

This directory contains a decoder-only transformer model that uses a Mixture of Experts (MoE) layer. The model architecture is similar to the one in the `subspace_decoder` directory, with both using `MultiheadLatentAttention`.

The primary difference is in the feed-forward network (FFN) block. While the `subspace_decoder` uses a standard dense FFN, this model implements a `SparseMoEFeedForward` layer. This layer uses a `NoisyTopKRouter` to dynamically route each token to a small subset of expert networks. This allows for a much larger number of parameters in the model, while keeping the computational cost for each token constant, as only a fraction of the experts are used for each input.

# To run training script
```
python -m moe.scripts.train --config moe/configs/gpt-2_sparse_moe_wiki103.json
```

## Technical papers implemented

@article{Shazeer2017Outrageously,
  title        = {Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  author       = {Noam Shazeer and Azalia Mirhoseini and Krzysztof Maziarz and Andy Davis and Quoc V. Le and Geoffrey E. Hinton and Jeff Dean},
  journal      = {arXiv preprint arXiv:1701.06538},
  year         = {2017},
  url          = {https://arxiv.org/abs/1701.06538}
}


@article{DeepSeekV3_2024,
  title        = {DeepSeek-V3 Technical Report},
  author       = {DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and …},
  journal      = {arXiv preprint arXiv:2412.19437},
  year         = {2024},
  url          = {https://arxiv.org/abs/2412.19437}
}

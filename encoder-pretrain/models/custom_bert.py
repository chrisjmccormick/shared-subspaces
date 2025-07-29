import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertLayer, BertSelfAttention

try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepSeekV3Attention
except Exception:
    DeepSeekV3Attention = None


class OutputSubspaceAttention(BertSelfAttention):
    """BertSelfAttention with an optional shared output projection."""

    def __init__(self, config):
        super().__init__(config)
        self.output_subspace = getattr(config, "output_subspace", False)
        self.output_subspace_dim = getattr(config, "output_subspace_dim", None)
        if self.output_subspace:
            dim = self.output_subspace_dim or config.hidden_size
            self.up_proj = nn.Linear(config.hidden_size, dim, bias=False)
            self.down_proj = nn.Linear(dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        out = super().forward(hidden_states, *args, **kwargs)[0]
        if self.output_subspace:
            out = self.down_proj(self.up_proj(out))
        return (out,)


class MLAAttention(OutputSubspaceAttention if DeepSeekV3Attention is None else DeepSeekV3Attention):
    """Multihead Latent Attention with optional shared output projection."""

    def __init__(self, config):
        super().__init__(config)
        if DeepSeekV3Attention is None:
            # fall back to standard attention with subspace
            pass
        self.output_subspace = getattr(config, "output_subspace", False)
        self.output_subspace_dim = getattr(config, "output_subspace_dim", None)
        if self.output_subspace:
            dim = self.output_subspace_dim or config.hidden_size
            self.up_proj = nn.Linear(config.hidden_size, dim, bias=False)
            self.down_proj = nn.Linear(dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        out = super().forward(hidden_states, *args, **kwargs)[0]
        if self.output_subspace:
            out = self.down_proj(self.up_proj(out))
        return (out,)


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.up = nn.Linear(in_features, rank, bias=False)
        self.down = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.down(self.up(x))


class CustomBertForMaskedLM(BertForMaskedLM):
    """BERT model with optional MLA, output subspace, and decomposed MLP."""

    @classmethod
    def from_config(cls, exp_cfg):
        cfg = BertConfig.from_pretrained(exp_cfg["model_name"])
        cfg.output_subspace = exp_cfg.get("output_subspace", False)
        cfg.output_subspace_dim = exp_cfg.get("output_subspace_dim", None)
        cfg.use_mla = exp_cfg.get("use_mla", False)
        cfg.ffn_decompose = exp_cfg.get("ffn_decompose", False)
        cfg.ffn_rank = exp_cfg.get("ffn_rank", None)
        return cls(cfg)

    def __init__(self, config):
        super().__init__(config)
        for layer in self.bert.encoder.layer:
            if getattr(config, "use_mla", False):
                layer.attention.self = MLAAttention(config)
            elif getattr(config, "output_subspace", False):
                layer.attention.self = OutputSubspaceAttention(config)
            if getattr(config, "ffn_decompose", False):
                rank = config.ffn_rank or config.intermediate_size
                layer.intermediate.dense = LowRankLinear(config.hidden_size, config.intermediate_size, rank)
                layer.output.dense = LowRankLinear(config.intermediate_size, config.hidden_size, rank)

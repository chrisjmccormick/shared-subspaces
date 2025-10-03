# `utils.py`

# Utility helpers for experiments

from typing import Iterable, Tuple
import torch.nn as nn


def format_size(num: int) -> str:
    """Return a human readable string for the given integer."""
    suffixes = [" ", "K", "M", "B"]
    base = 1024
    for suffix in suffixes:
        if abs(num) < base:
            if num % 1 != 0:
                return f"{num:.2f}{suffix}"
            else:
                return f"{num:.0f}{suffix}"
        num /= base
    if num % 1 != 0:
        return f"{num:.2f}T"
    return f"{num:.0f}T"


def summarize_parameters(model: nn.Module, display_bias: bool = True) -> int:
    """Print a table of parameter names, shapes and counts."""
    params: Iterable[Tuple[str, nn.Parameter]] = list(model.named_parameters())
    print("The model has {:} different named parameters.\n".format(len(params)))

    total_params = 0
    for _, p in params:
        total_params += p.numel()

    print(f"Total elements: {format_size(total_params)}\n")
    print(
        "Parameter Name                                              Dimensions       Total Values    Trainable\n"
    )

    for p_name, p in params:
        p_size = list(p.size())
        for i in range(len(p_size) - 1, -1, -1):
            if p_size[i] == 1:
                del p_size[i]
        if len(p_size) == 1:
            if not display_bias:
                continue
            p_dims = "{:>10,} x {:<10}".format(p.size()[0], "-")
        elif len(p_size) == 2:
            p_dims = "{:>10,} x {:<10,}".format(p.size()[0], p.size()[1])
        elif len(p_size) == 3:
            p_dims = "{:>10,} x {:,} x {:<10}".format(p.size()[0], p.size()[1], p.size()[2])
        elif len(p_size) == 4:
            p_dims = "{:>10,} x {:,} x {:,} x {:<10}".format(
                p.size()[0], p.size()[1], p.size()[2], p.size()[3]
            )
        else:
            print("Unexpected: ", p.size(), p_name)
            break
        print(
            "{:<55} {:}    {:>6}    {:}".format(
                p_name, p_dims, format_size(p.numel()), p.requires_grad
            )
        )

    print(f"\nTotal elements: {format_size(total_params)}\n")
    return total_params

"""`def make_shorthand`"""

def make_shorthand(model_cfg):
    """
    Takes an instance subencoder `*Config` and constructs a shorthand
    name for the model based on settings.
    """

    dense_str = str(model_cfg.num_dense_layers) + "mha + "

    if model_cfg.output_subspace:
        o_str = "." + str(model_cfg.o_latent_dim)
    else:
        o_str = ""

    # If no output subspace is used, the dimension will show as -1.
    attn_str = (
        dense_str
        + "mla."
        + str(model_cfg.q_latent_dim)
        + "."
        + str(model_cfg.kv_latent_dim)
        + o_str
    )

    # MLP Configuration
    if model_cfg.ffn_decompose:
        dense_str = (
            str(model_cfg.num_dense_layers)
            + "mlp."
            + str(model_cfg.intermediate_size)
            + " + "
        )

        mlp_str = (
            dense_str
            + str(model_cfg.num_hidden_layers - model_cfg.num_dense_layers)
            + "dcmp."
            + "x"
            + str(model_cfg.intermediate_size)
            + "."
            + str(model_cfg.ffn_rank)
        )
    else:
        mlp_str = "mlp." + str(model_cfg.intermediate_size)

    # Assemble string
    shorthand = (
        f"{attn_str} - {mlp_str} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers}"
    )

    """
    The run name includes training settings

    run_name = (
        f"{config['stats']['total_elements']} - "
        f"{attn_str} - {mlp_str} - "
        f"h{model_cfg.hidden_size} - l{model_cfg.num_hidden_layers} - "
        f"bs{ptrain_cfg['train_batch_size']} - lr{lr_str} - "
        f"seq{ptrain_cfg['max_seq_length']}"
    )
    """

    return shorthand

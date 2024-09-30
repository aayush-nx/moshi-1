# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.compile import torch_compile_lazy


@torch_compile_lazy
def gating_forward_kernel(
    weight_in: torch.Tensor, weight_out: torch.Tensor, activation, x: torch.Tensor
):
    """
    Applies gating mechanism to the input tensor.

    Args:
        weight_in (torch.Tensor): Input weight matrix for the first linear transformation.
        weight_out (torch.Tensor): Output weight matrix for the final linear transformation.
        activation (callable): Activation function to be applied.
        x (torch.Tensor): Input tensor of shape [B, T, D] where B is batch size, T is sequence length, and D is input dimension.

    Returns:
        torch.Tensor: Output tensor after applying gating mechanism, of shape [B, T, D_out] where D_out is the output dimension.

    The function performs the following steps:
    1. Applies a linear transformation to the input.
    2. Reshapes the result to separate the gating and content components.
    3. Applies the activation function to the gating component and multiplies it element-wise with the content component.
    4. Applies a final linear transformation to produce the output.
    """
    x = F.linear(x, weight_in)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = F.linear(x, weight_out)
    return x


class ActivationGating(nn.Module):
    """
    Implements a gating mechanism for feed-forward networks (FFN) in transformers.
    
    This module applies a gating operation to the input, which involves:
    1. Expanding the input dimension
    2. Splitting the expanded tensor into two parts: one for gating and one for content
    3. Applying an activation function to the gating part
    4. Element-wise multiplication of the activated gate with the content
    5. Projecting the result back to the original dimension

    Args:
        dim (int): Dimension of the input and output tensors.
        dim_feedforward (int): Dimension of the intermediate (expanded) representation.
        activation (callable): Activation function to apply to the gating mechanism.
        **factory_kwargs: Additional keyword arguments for the linear layers (e.g., device, dtype).

    Attributes:
        linear_in (nn.Linear): Input linear transformation that expands the input.
        linear_out (nn.Linear): Output linear transformation that projects back to the original dimension.
        activation (callable): Activation function for the gating mechanism.
    """

    _fsdp_final = True  # Indicates this is the final module in a Fully Sharded Data Parallel setup

    def __init__(self, dim: int, dim_feedforward: int, activation, **factory_kwargs):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        
        # Calculate the hidden dimension
        if dim_feedforward == 4 * dim:
            # Special case: if dim_feedforward is 4 times the input dimension,
            # use a specific ratio (21/8) for the hidden dimension
            hidden = (21 * dim) // 8
        else:
            # Otherwise, use 2/3 of dim_feedforward
            hidden = (2 * dim_feedforward) // 3
        
        # Input linear layer: expands from 'dim' to '2 * hidden'
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        
        # Output linear layer: projects from 'hidden' back to 'dim'
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        
        # Store the activation function
        self.activation = activation

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ActivationGating module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, dim], where B is batch size and T is sequence length.

        Returns:
            torch.Tensor: Output tensor of shape [B, T, dim] after applying the gating mechanism.
        """
        return gating_forward_kernel(
            self.linear_in.weight, self.linear_out.weight, self.activation, x
        )


def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(torch.nn.functional, name)
    elif name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def _make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    return ActivationGating(
        dim, dim_feedforward, _get_activation(name), **factory_kwargs
    )


def make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    gating = _make_gating(name, dim, dim_feedforward, **factory_kwargs)
    max_params = 2 * dim * dim_feedforward
    params = sum(p.numel() for p in gating.parameters())
    assert (
        params <= max_params
    ), f"{name} gating has {params} params, max is {max_params}"
    return gating

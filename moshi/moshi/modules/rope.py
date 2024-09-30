# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import math
import torch
from ..utils.compile import torch_compile_lazy


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    time_before_heads: bool = False,
):
    """
    Apply Rotary Positional Embedding (RoPE) to query and key tensors.

    This function applies a rotation to the query and key tensors based on their position,
    which helps the model learn relative positions in the input sequence.

    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]` if time_before_heads else `[B, H, T, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]` if time_before_heads else `[B, H, T, D]`.
        offset (torch.Tensor): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin. Default is 10,000.
        time_before_heads (bool): if True, expected [B, T, H, D], else [B, H, T, D]. Default is False.

    Returns:
        tuple: (rotated_q, rotated_k), each with the same shape as the input q and k.
    """

    # Determine shape based on time_before_heads flag
    if time_before_heads:
        B, T, H, D = q.shape  # [Batch, Time, Heads, Dimension]
    else:
        B, H, T, D = q.shape  # [Batch, Heads, Time, Dimension]
    assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    # Generate frequency bands
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))  # Shape: [D/2]

    # Generate time steps
    ts = offset.float() + torch.arange(T, device=q.device, dtype=torch.float32)  # Shape: [T]
    if time_before_heads:
        ts = ts.view(-1, 1, 1)  # Shape: [T, 1, 1]
    else:
        ts = ts.view(1, -1, 1)  # Shape: [1, T, 1]

    # Reshape q and k for rotation
    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)  # Shape: [..., D/2, 2]
    k = k.view(*dims, D // 2, 2)  # Shape: [..., D/2, 2]

    # Split real and imaginary parts
    qr, qi = q[..., 0].float(), q[..., 1].float()  # Shape: [..., D/2]
    kr, ki = k[..., 0].float(), k[..., 1].float()  # Shape: [..., D/2]

    # Compute rotation
    rotr = torch.cos(freqs * ts)  # Shape: [T, 1, D/2] or [1, T, D/2]
    roti = torch.sin(freqs * ts)  # Shape: [T, 1, D/2] or [1, T, D/2]

    # Apply rotation
    qor = qr * rotr - qi * roti  # Shape: [..., D/2]
    qoi = qr * roti + qi * rotr  # Shape: [..., D/2]

    kor = kr * rotr - ki * roti  # Shape: [..., D/2]
    koi = kr * roti + ki * rotr  # Shape: [..., D/2]

    # Combine real and imaginary parts and convert back to original dtype
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)  # Shape: [..., D/2, 2]
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)  # Shape: [..., D/2, 2]

    # Reshape to original dimensions
    return qo.view(*dims, D), ko.view(*dims, D)
    return qo.view(*dims, D), ko.view(*dims, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offset, self.max_period, time_before_heads)

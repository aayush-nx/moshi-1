# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, + CUDA Graphable.
Optimized for inference.

See `StreamingTransformer` for more information.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils.compile import no_compile
from .gating import make_gating
from .rope import RotaryEmbedding
from .streaming import StreamingModule, StreamingContainer


class LayerNormF32(nn.LayerNorm):
    """
    Layer normalization that casts inputs to float32 for improved precision.

    This class extends nn.LayerNorm to perform normalization in float32 precision,
    regardless of the input tensor's dtype. This can help maintain numerical
    stability, especially for mixed precision training.

    Shape:
        - Input: (*, C) where * is any number of dimensions and C is the number of channels.
        - Output: (*, C), same shape as the input.

    Returns:
        torch.Tensor: The layer normalized tensor in the same dtype as the input.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()  # Cast to float32
        out_f32 = super().forward(x_f32)  # Perform normalization in float32
        return out_f32.to(input.dtype)  # Cast back to original dtype


def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: tp.Optional[torch.dtype],
    eps: float,
) -> torch.Tensor:
    """
    Applies Root Mean Square (RMS) Normalization to the input tensor.

    RMS Norm Formula:
    y = (x / sqrt(mean(x^2) + eps)) * alpha

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, C) where B is batch size,
                          T is sequence length, and C is the number of channels.
        alpha (torch.Tensor): Learnable scale parameter of shape (1, 1, C).
        dtype (torch.dtype, optional): Data type to use for intermediate computations.
        eps (float): Small constant added to the variance for numerical stability.

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input (B, T, C).

    Raises:
        AssertionError: If input tensor is not 3-dimensional.

    Note:
        The mean is computed across the T dimension for each channel independently,
        resulting in a tensor of shape (B, 1, C) which is then broadcast during normalization.
    """
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    """
    Root Mean Square (RMS) Normalization layer.

    This layer applies RMS normalization to the input tensor, which normalizes the
    activations of the layer for each sample in a batch using the RMS value.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create sinusoidal positional embedding for transformer models.

    This function generates a sinusoidal positional embedding tensor that can be used
    to inject position information into transformer models. The embedding is created
    using a combination of sine and cosine functions with different frequencies.

    Args:
        positions (torch.Tensor): LongTensor of positions. Shape: [B, T]
        dim (int): Dimension of the embedding (must be even).
        max_period (float): Maximum period of the cosine/sine functions. Default: 10000
        dtype (torch.dtype): Data type to use for the embedding. Default: torch.float32

    Returns:
        torch.Tensor: Sinusoidal positional embedding. Shape: [B, T, dim]

    Note:
        - The function assumes a batch-time-channel (BTC) format for the output.
        - The dimension (dim) must be even to allow for equal split between sine and cosine.
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    half_dim = dim // 2
    positions = positions.to(dtype)  # Shape: [B, T]
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)  # Shape: [1, 1, half_dim]
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # Scalar tensor to avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))  # Shape: [B, T, half_dim]
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)  # Shape: [B, T, dim]


def multi_linear(
    num_linear: int,
    weight: torch.Tensor,
    x: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """
    Apply a multi-linear layer to the given input, where each time step uses a different set of weights.

    This function implements a time-dependent linear transformation, where each time step
    has its own set of weights. It's particularly useful in scenarios where the linear
    transformation needs to vary across time steps, such as in certain types of
    attention mechanisms or time-varying neural networks.

    Args:
        num_linear (int): Total number of linear transformations (typically equal to
                          the maximum sequence length).
        weight (torch.Tensor): Weight tensor containing all linear transformations.
        x (torch.Tensor): Input tensor to transform.
        offset (int): Starting index for selecting weights, useful for sequential
                      processing or when continuing from a previous state.

    Returns:
        torch.Tensor: Transformed output tensor.

    Note:
        The function assumes that `num_linear` is at least as large as `T + offset`
        to ensure there are enough weights for all time steps.
    """
    B, T, C = x.shape  # x: [B, T, chin]
    ys = []
    chout, chin = weight.shape  # weight: [num_linear * chout, chin]
    weight = weight.view(num_linear, -1, chin)  # weight: [num_linear, chout, chin]
    for t in range(T):
        y = F.linear(x[:, t], weight[t + offset])  # y: [B, chout]
        ys.append(y)
    out = torch.stack(ys, 1)  # out: [B, T, chout]
    return out


def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    """
    Deactivates or changes the context span (in time steps) for all StreamingMultiheadAttention modules in a model.

    This function traverses the entire model and sets the 'context' attribute of all
    StreamingMultiheadAttention modules to the specified value. This allows for dynamic
    adjustment of the attention span in streaming scenarios.

    Args:
        model (nn.Module): The PyTorch model to modify.
        context (int or None): New context value to set. If None, it effectively
                               deactivates the context limitation.

    Returns:
        None

    Note:
        This is not a context manager but a plain function that permanently changes the context.
        It was initially designed as a context manager, but this led to inconsistencies
        between forward and backward passes when using activation checkpointing.

    Example:
        >>> model = MyTransformerModel()
        >>> set_attention_context(model, context=100)  # Set context to 100 time steps
        >>> set_attention_context(model, context=None)  # Remove context limitation
    """
    for module in model.modules():  # Iterate through all modules in the model
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context  # Set the context attribute of StreamingMultiheadAttention modules


class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor  # Shape: [B, H, T, D]
    values: torch.Tensor  # Shape: [B, H, T, D]
    positions: torch.Tensor  # Shape: [T]

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        """
        Create a KVCacheResult from given keys and values tensors.

        This method constructs a KVCacheResult object, which includes the keys,
        values, and corresponding positions. It performs shape validation and
        generates position indices.

        Args:
            keys (torch.Tensor): The key tensor. Shape: [B, H, T, D]
            values (torch.Tensor): The value tensor. Shape: [B, H, T, D]

        Returns:
            KVCacheResult: A named tuple containing keys, values, and positions.

        Raises:
            AssertionError: If the shapes of keys and values are incompatible.
        """
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T), "Values shape must match keys shape except for last dimension"
        positions = torch.arange(T, device=keys.device, dtype=torch.long)  # Shape: [T]
        return KVCacheResult(keys, values, positions)


class RingKVCache:
    """
    Efficient streaming KVCache compatible with CUDA Graph.

    This class implements a ring buffer for key-value caching in transformer models,
    allowing for efficient streaming inference.

    Args:
        batch_size (int): Number of sequences in a batch.
        num_heads (int): Number of attention heads.
        dim_per_head (int): Dimension of each attention head.
        capacity (int): Maximum number of time steps to store in the cache.
        device (torch.device): Device on which to initialize the cache. Default is CUDA.
        dtype (torch.dtype): Data type for the cache. Default is bfloat16.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        #self.cache Shape: [2, B, H, T, D] where 2 is for keys and values
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        
        # Tracks the end of the valid data in the cache
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self):
        """Resets the cache by zeroing out the end offset."""
        self.end_offset.zero_()

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        """
        Updates the cache with new keys and values and returns the complete cache.

        Args:
            k (torch.Tensor): New keys to add. Shape: [B, H, T, D]
            v (torch.Tensor): New values to add. Shape: [B, H, T, D]

        Returns:
            KVCacheResult: Named tuple containing the complete keys, values, and their positions.
        """
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        # Calculate indices for the new entries
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        # Update the cache with new keys and values
        self.cache[0].index_copy_(2, indexes, k)  # Update keys
        self.cache[1].index_copy_(2, indexes, v)  # Update values
        self.end_offset.add_(T)

        keys = self.cache[0]    # Shape: [B, H, T, D]
        values = self.cache[1]  # Shape: [B, H, T, D]

        # Generate position information
        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )
        invalid = indexes >= self.end_offset

        end_index = self.end_offset % self.capacity
        delta = indexes - end_index

        # If last key is for step S, and capacity is C, last key was written at index S % C.
        # then end_offset = S + 1, and end_index = (S + 1) % C.
        # Then for index = (S % C), delta = -1, and the next code gives us:
        # position(index) = (S + 1) - 1 = S, all good.
        # Now the time step at end_offset is actually the oldest in the KVCache, e.g., its
        # position should be (S - self.capacity + 1).
        # The following code gives us:
        # position(index + 1) = S + 1 + 0 - self.capacity.

        positions = torch.where(
            delta <= 0,
            self.end_offset + delta,
            self.end_offset + delta - self.capacity,
        )
        # Mark invalid positions (those not yet filled) with -1
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


@dataclass
class _MHAState:
    kv_cache: RingKVCache
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    This class implements a streaming version of multi-head attention, which can be used
    for causal language modeling tasks. It supports both standard and streaming inference modes.

    Args:
        embed_dim (int): Dimension of the input and output embeddings.
        num_heads (int): Number of attention heads.
        causal (bool): If True, applies a causal mask to the attention. Default is False.
        context (int, optional): Number of time steps the attention can access.
            For causal attention, it's the number of past steps. For non-causal,
            it's split evenly between past and future. Default is None (unlimited context).
        rope (`RotaryEmbedding`, optional): Rotary position embedding to use. Default is None.
        weights_per_step (int): Number of unique weight sets to use per time step.
            If non-zero, uses different weights for each possible time step. Default is 0.
        device (torch.device, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): Data type to use for module parameters.

    Shape:
        - Input: `(batch_size, seq_len, embed_dim)`
        - Output: `(batch_size, seq_len, embed_dim)`

    Attributes:
        in_proj_weight (nn.Parameter): Combined weight for query, key, and value projections.
        in_proj_bias (nn.Parameter): Combined bias for query, key, and value projections.
        out_proj (nn.Linear): Output projection layer.
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        mult = 1
        self.weights_per_step = weights_per_step
        if weights_per_step:
            mult = weights_per_step
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = in_proj.weight  # Shape: [mult * 3 * embed_dim, embed_dim]
        self.in_proj_bias = in_proj.bias  # Shape: [mult * 3 * embed_dim] or None
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )  # out_proj.weight shape: [mult * embed_dim, embed_dim]

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        """Initialize the streaming state for the attention module.

        Args:
            batch_size (int): Batch size for the input.

        Returns:
            _MHAState: Initialized streaming state.

        Raises:
            RuntimeError: If context is None and weights_per_step is 0.
        """
        if self.context is None:
            if self.weights_per_step:
                capacity = self.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create a streaming KVCache without a context to estimate capacity."
                )
        else:
            capacity = self.context
        device = self.in_proj_weight.device
        # TODO: the following estimation will not work great with FSDP.
        dtype = self.in_proj_weight.dtype
        dim_per_head = self.embed_dim // self.num_heads
        kv_cache = RingKVCache(
            batch_size, self.num_heads, dim_per_head, capacity, device, dtype
        )
        return _MHAState(
            kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v) -> KVCacheResult:
        """Complete the key and value tensors using the KV cache.

        Args:
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            KVCacheResult: Completed key and value tensors with position information.
        """
        state = self._streaming_state
        if state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Forward pass of the StreamingMultiheadAttention module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        state = self._streaming_state
        T = query.shape[1]  # T: sequence length

        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)  # Shape: (1,)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal"
            offset = state.offset  # Shape: (1,)
            offset_cpu = state.offset_cpu

        if self.weights_per_step:
            projected = multi_linear(
                self.weights_per_step, self.in_proj_weight, query, offset_cpu
            )  # Shape: (batch_size, T, 3 * embed_dim * mult)
        else:
            projected = nn.functional.linear(query, self.in_proj_weight)  # Shape: (batch_size, T, 3 * embed_dim * mult)
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )  # Shape of each: (batch_size, num_heads, T, dim_per_head)

        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)  # Shapes unchanged

        k, v, pos_k = self._complete_kv(k, v)  # k, v shapes: (batch_size, num_heads, T', dim_per_head), pos_k shape: (T')
        if self.causal:
            pos_k = pos_k.view(1, -1)  # Shape: (1, T')
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(
                -1, 1
            )  # Shape: (T, 1)
            delta = pos_q - pos_k  # Shape: (T, T')
            attn_bias = (pos_k >= 0) & (delta >= 0)  # Shape: (T, T')
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)  # Shape: (T, T')
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)  # Shape: (batch_size, num_heads, T, dim_per_head)

        x = rearrange(x, "b h t d -> b t (h d)")  # Shape: (batch_size, T, embed_dim)
        if self.weights_per_step:
            x = multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)  # Shape: (batch_size, T, embed_dim)
        else:
            x = self.out_proj(x)  # Shape: (batch_size, T, embed_dim * mult)
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        return x  # Shape: (batch_size, T, embed_dim * mult)


@dataclass
class _LayerState:
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0


class StreamingTransformerLayer(StreamingModule[_LayerState]):
    """TransformerLayer with Streaming / Causal support.

    This class implements a single layer of a streaming transformer, supporting causal
    and non-causal attention with optional context limitation. It includes self-attention
    and feed-forward components, with various normalization and gating options.

    Args:
        d_model (int): Dimension of the model (input and output).
        num_heads (int): Number of attention heads.
        dim_feedforward (int | list[int]): Dimension(s) of the feedforward network.
            If a list, must match weights_per_step.
        causal (bool): If True, applies causal masking to the attention. Default is False.
        context (int, optional): Maximum context size for attention. If None, unlimited.
        rope (`RotaryEmbedding`, optional): Rotary position embedding to use.
        norm (str): Type of normalization to use. Currently only 'layer_norm' is supported.
        layer_scale (float, optional): Initial scale for LayerScale. If None, LayerScale is not used.
        gating (str): Type of gating mechanism for the feedforward network.
        weights_per_step (int): Number of unique weight sets to use per time step.
        activation (callable): Activation function for the feedforward network.
        skip_self_attn (bool): If True, skips the self-attention module and its normalization.
        device (torch.device, optional): Device on which to initialize the layer.
        dtype (torch.dtype, optional): Data type to use for layer parameters.

    Shape:
        - Input: `(batch_size, seq_len, d_model)`
        - Output: `(batch_size, seq_len, d_model)`
    """

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {weights_per_step}"
            )
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block of the transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        state = self._streaming_state
        offset = 0
        if state is not None:
            offset = state.offset_cpu
        x_orig = x  # Shape: (batch_size, seq_len, d_model)
        x = self.norm2(x)  # Shape: (batch_size, seq_len, d_model)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            # Shape: (batch_size, seq_len, dim_feedforward)
            hidden = self.activation(self.linear1(x))
            # Shape: (batch_size, seq_len, d_model)
            update = self.linear2(hidden)
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                B, T, D = x.shape
                ys = []
                for t in range(T):
                    # Shape: (batch_size, 1, d_model)
                    y = self.gating[offset + t](x[:, t : t + 1])
                    ys.append(y)
                # Shape: (batch_size, seq_len, d_model)
                update = torch.cat(ys, dim=1)
            else:
                # Shape: (batch_size, seq_len, d_model)
                update = self.gating(x)
        # Shape: (batch_size, seq_len, d_model)
        return x_orig + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention block of the transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.skip_self_attn:
            return x
        x_orig = x  # Shape: (batch_size, seq_len, d_model)
        x = self.norm1(x)  # Shape: (batch_size, seq_len, d_model)
        # Shape: (batch_size, seq_len, d_model)
        update = self.self_attn(x, x, x)
        # Shape: (batch_size, seq_len, d_model)
        return x_orig + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StreamingTransformerLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        with ExitStack() as stack:
            if x.device.type != 'cuda':
                stack.enter_context(no_compile())
            x = self._sa_block(x)  # Shape: (batch_size, seq_len, d_model)
            x = self._ff_block(x)  # Shape: (batch_size, seq_len, d_model)
            state = self._streaming_state
            if state:
                state.offset_cpu += x.shape[1]
            return x  # Shape: (batch_size, seq_len, d_model)


@dataclass
class _TransformerState:
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()


class StreamingTransformer(StreamingModule[_TransformerState]):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, sin_rope, or none).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `StreamingTransformerLayer`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offset.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            state.offset.add_(T)
        return x


class ProjectedTransformer(StreamingContainer):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(self, x, *args, **kwargs):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys

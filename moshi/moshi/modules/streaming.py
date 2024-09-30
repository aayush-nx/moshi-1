# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Streaming module API that should be implemented by all Streaming components,
"""

import abc
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import math
import typing as tp
from torch import nn
import torch


class Resetable(tp.Protocol):
    """Protocol for objects that can be reset to their initial state."""

    def reset(self) -> None:
        """Reset the object to its initial state."""
        pass


# Type variable for streaming state, constrained to be Resetable
State = tp.TypeVar("State", bound=Resetable)


class StreamingModule(abc.ABC, nn.Module, tp.Generic[State]):
    """Common API for streaming components.

    Each streaming component has a streaming state, which is just a dict[str, Tensor].
    By convention, the first dim of each tensor must be the batch size.
    Don't use dots in the key names, as this would clash with submodules
    (like in state_dict).

    If `self._is_streaming` is True, the component should use and remember
    the proper state inside `self._streaming_state`.

    To set a streaming component in streaming state, use

        with module.streaming():
            ...

    This will automatically reset the streaming state when exiting the context manager.
    This also automatically propagates to all streaming children module.

    Some module might also implement the `StreamingModule.flush` method, although
    this one is trickier, as all parents module must be StreamingModule and implement
    it as well for it to work properly. See `StreamingSequential` after.

    This class provides methods for managing streaming state, including
    initializing, starting, stopping, resetting, and propagating streaming
    behavior to child modules. It also includes utilities for getting and
    setting the complete streaming state of a module hierarchy.
    """

    def __init__(self) -> None:
        """Initialize the StreamingModule."""
        super().__init__()
        self._streaming_state: State | None = None
        self._streaming_propagate: bool = True

    @property
    def is_streaming(self):
        """Check if the module is in streaming mode."""
        return self._streaming_state is not None

    def set_streaming_propagate(self, streaming_propagate: bool):
        """Set whether streaming should propagate to child modules."""
        self._streaming_propagate = streaming_propagate

    def _apply_named_streaming(self, fn: tp.Any):
        """Apply a function to all streaming modules in the hierarchy."""
        def _handle_module(prefix: str, module: nn.Module, recurse: bool = True):
            propagate = True
            if isinstance(module, StreamingModule):
                if module._streaming_propagate:
                    fn(prefix, module)
                else:
                    propagate = False
            if not recurse:
                return
            if propagate:
                for name, child in module.named_children():
                    _handle_module(prefix + "." + name, child)

        _handle_module("", self, recurse=False)
        for name, child in self.named_children():
            _handle_module(name, child)

    def _start_streaming(self, batch_size: int):
        """Start streaming mode for all modules in the hierarchy."""
        def _start_streaming(name: str, module: StreamingModule):
            module._streaming_state = module._init_streaming_state(batch_size)

        self._apply_named_streaming(_start_streaming)

    def _stop_streaming(self):
        """Stop streaming mode for all modules in the hierarchy."""
        def _stop_streaming(name: str, module: StreamingModule):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming)

    @abc.abstractmethod
    def _init_streaming_state(self, batch_size: int) -> State: ...

    def streaming_forever(self, batch_size: int):
        """Start streaming mode indefinitely."""
        self._start_streaming(batch_size)

    @contextmanager
    def streaming(self, batch_size: int):
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    def reset_streaming(self):
        """Reset the streaming state for all modules in the hierarchy."""
        def _reset(name: str, module: StreamingModule):
            state = module._streaming_state
            if state is None:
                raise ValueError(
                    f"Trying to reset streaming, but {name} wasn't streaming."
                )
            state.reset()

        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> dict[str, tp.Any]:
        """Return the complete streaming state, including that of sub-modules."""
        state: dict[str, tp.Any] = {}

        def _add(name: str, module: StreamingModule):
            state[name] = module._streaming_state

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: dict[str, tp.Any]):
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: StreamingModule):
            if name in state:
                module._streaming_state = state[name]
                state.pop(name)
            else:
                raise RuntimeError(f"Expected to find a streaming state for {name}.")

        self._apply_named_streaming(_set)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")


@dataclass
class _NullState:
    """A null state class for streaming modules that don't require state."""
    pass

    def reset(self) -> None:
        """Reset method (no-op for null state)."""
        pass


class StreamingContainer(StreamingModule[_NullState]):
    """
    A streaming container that doesn't require any state.
    This class is used as a base for streaming modules that don't need to maintain
    any internal state between calls. It initializes with a null state (_NullState)
    and can be used to wrap other streaming modules or as a parent class for
    stateless streaming components.
    """

    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()


@dataclass
class _StreamingAddState:
    """
    State for StreamingAdd module.
    Stores previous tensors for x and y to handle streaming addition.
    """
    previous_x: torch.Tensor | None = None
    previous_y: torch.Tensor | None = None

    def reset(self):
        """Reset the state by setting previous tensors to None."""
        self.previous_x = None
        self.previous_y = None


class StreamingAdd(StreamingModule[_StreamingAddState]):
    """
    A streaming module that performs element-wise addition of two tensors.
    
    This module handles streaming addition by maintaining state for partial
    inputs and aligning tensors of different lengths. It supports both
    streaming and non-streaming modes of operation.
    """

    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        """Initialize the streaming state for the StreamingAdd module."""
        return _StreamingAddState()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform element-wise addition of two tensors, handling streaming and non-streaming cases.

        In non-streaming mode, simply adds x and y.
        In streaming mode, concatenates with previous partial inputs, aligns tensors,
        updates the streaming state, and returns the sum of aligned portions.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Result of element-wise addition.
        """
        if self._streaming_state is None:
            return x + y
        else:
            prev_x = self._streaming_state.previous_x
            prev_y = self._streaming_state.previous_y
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            m_l = min(x.shape[-1], y.shape[-1])
            self._streaming_state.previous_x = x[..., m_l:]
            self._streaming_state.previous_y = y[..., m_l:]
            return x[..., :m_l] + y[..., :m_l]


@dataclass
class _StreamingConvState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingConv1d(nn.Conv1d, StreamingModule[_StreamingConvState]):
    """
    A streaming 1D convolution layer that supports both streaming and non-streaming modes.

    This class extends nn.Conv1d and implements the StreamingModule interface to handle
    streaming convolution operations. It maintains a state of previous inputs to ensure
    correct convolution across streaming chunks.

    Args:
        *args: Variable length argument list for nn.Conv1d.
        **kwargs: Arbitrary keyword arguments for nn.Conv1d.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        """Initialize the streaming state for the convolution layer."""
        return _StreamingConvState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the streaming convolution.

        In non-streaming mode, it behaves like a regular Conv1d.
        In streaming mode, it handles partial inputs and maintains state across calls.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor after convolution.

        Note:
            In streaming mode, this method uses the `previous` attribute of the
            streaming state to store and retrieve information from previous calls.
            The `previous` tensor contains the tail end of the input from the last
            call, which is prepended to the current input to ensure continuity
            in the convolution operation across chunk boundaries.
        """
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # Compute the number of full convolution frames ready to be processed
            # This formula allows for one output frame when input length equals kernel size
            # For a more conservative approach (input > kernel size):
            # num_frames = max(0, int(math.floor((T - kernel - 1) / stride) + 1))
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            

            # Calculate the offset in the input tensor up to which data has been processed
            # offset represents the position beyond which data will be used in future computations
            # Data before offset will not be used again, as we advance by 'stride' for each frame
            offset = num_frames * stride
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                # - (num_frames - 1) * stride calculates the starting position of the last frame.
                # - Adding kernel ensures we include all the elements needed for the last frame.
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.out_channels, 0, device=input.device, dtype=input.dtype
                )
            return out


@dataclass
class _StreamingConvTrState:
    partial: torch.Tensor | None = None

    def reset(self):
        self.partial = None


class RawStreamingConvTranspose1d(
    nn.ConvTranspose1d, StreamingModule[_StreamingConvTrState]
):
    """
    A streaming version of ConvTranspose1d that can process input in chunks.

    This class extends nn.ConvTranspose1d and implements the StreamingModule interface.
    It allows for efficient processing of input in a streaming fashion, maintaining
    internal state between calls to handle partial results.

    Args:
        *args: Variable length argument list passed to nn.ConvTranspose1d.
        **kwargs: Arbitrary keyword arguments passed to nn.ConvTranspose1d.

    Attributes:
        _streaming_state (_StreamingConvTrState): Stores the partial results between calls.
            The 'partial' field in this state represents the overlapping output
            that needs to be combined with the next chunk's output.

    Note:
        - Padding should be handled outside this module (padding[0] must be 0).
        - Dilation is not supported (must be 1).
        - Stride must be less than or equal to kernel_size.
        - Output padding is not supported (must be 0).
        - The 'partial' results are crucial for maintaining continuity between
          chunks in streaming mode, ensuring correct output reconstruction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
        """Initialize the streaming state."""
        return _StreamingConvTrState()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Forward pass of the streaming transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor after transposed convolution.

        Note:
            When in streaming mode, this method handles partial results and
            maintains the internal state between calls.
        """
        B, C, T = x.shape
        stride = self.stride[0]
        kernel = self.kernel_size[0]
        if self._streaming_state is None:
            return super().forward(x)
        else:
            if T == 0:
                return torch.empty(
                    B, self.out_channels, 0, device=x.device, dtype=x.dtype
                )
            out = super().forward(x)
            OT = out.shape[-1]
            partial = self._streaming_state.partial
            if partial is not None:
                # Due to the potential overlap, the rightmost output of the conv transpose is not
                # ready to be output, as it will receive contributions from the next input frames.
                # Here we recover those `partial` output frames. We know that the first time step
                # of the `partial` tensor corresponds to the first time step of `out` as anything
                # coming before the first time step of `out` would have been already flushed.
                PT = partial.shape[-1]
                if self.bias is not None:
                    out[..., :PT] += partial - self.bias[:, None]
                else:
                    out[..., :PT] += partial
            # The input is T, the output is S * (T - 1) + K.
            # The offset of the left of the next frame will be S * T
            # so everything between 0 and S * T is ready to be output, and we need
            # to keep in the internal state everything beyond that, i.e. S (T - 1) + K - S T = K - S
            invalid_steps = kernel - stride
            partial = out[..., OT - invalid_steps :]
            out = out[..., : OT - invalid_steps]
            self._streaming_state.partial = partial
            return out


def test():
    """
    Test function to validate the behavior of RawStreamingConv1d and RawStreamingConvTranspose1d.

    This function tests various combinations of kernel sizes, strides, and input lengths
    for both streaming and non-streaming modes. It ensures that the streaming implementation
    produces the same results as the non-streaming version within a small tolerance.

    The test covers:
    - Different kernel sizes and strides
    - Various input lengths
    - Both CPU and CUDA (if available) computations
    - Streaming mode with different chunk sizes

    Assertions are used to verify the correctness of shapes and output values.
    """
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        # Avoid the cuda optimizations that would take place on single precision
        # floats for convolutions.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    kernel_sizes = [1, 3, 4, 8, 15, 16]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chin = 6
    chout = 12

    for kernel, stride in itertools.product(kernel_sizes, strides):
        if stride > kernel:
            continue
        conv = RawStreamingConv1d(chin, chout, kernel, stride).to(device)
        convtr = RawStreamingConvTranspose1d(chout, chin, kernel, stride).to(device)

        for length in [4, 8, 32, 54, 65, 128, 1043]:
            print(f"ksize {kernel} strides {stride} len {length}")
            if length < kernel:
                continue
            batch_size = 3
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            for chunk_size in [1, 3, 5, 8]:
                ys = []
                zs = []
                with conv.streaming(batch_size), convtr.streaming(batch_size):
                    for offset in range(0, length, chunk_size):
                        chunk = x[..., offset : offset + chunk_size]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                y = y[..., : y_stream.shape[-1]]
                z = z[..., : z_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-6, delta
                num_frames = int((length - kernel) / stride) + 1
                assert num_frames == y_stream.shape[-1]

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()

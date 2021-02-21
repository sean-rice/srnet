from typing import Optional, Union

from detectron2.layers.shape_spec import ShapeSpec
from fvcore.nn import weight_init
import torch
import torch.nn

from .block_base import BLOCK_LAYERS_REGISTRY, BlockLayerBase

__all__ = ["FullyConnectedBlock", "DoubleFullyConnectedBlock"]


@BLOCK_LAYERS_REGISTRY.register()
class FullyConnectedBlock(BlockLayerBase):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        norm: Optional[Union[str, torch.nn.Module]] = "",
        activation: Optional[Union[str, torch.nn.Module]] = "relu",
        shortcut: bool = False,
    ):
        super(BlockLayerBase, self).__init__()

        if shortcut == True and n_inputs != n_outputs:
            raise ValueError(f"can't use shortcut when {n_inputs=} != {n_outputs=}.")
        self.shortcut: bool = shortcut

        norm_module: torch.nn.Module
        if isinstance(norm, torch.nn.Module):
            norm_module = norm
        elif isinstance(norm, str):
            norm_module = {"": torch.nn.Identity, "bn": torch.nn.BatchNorm1d,}[norm](
                n_inputs
            )
        elif norm is None:
            norm_module = torch.nn.Identity()
        else:
            raise ValueError(f"unknown value for norm {norm}.")

        act_module: torch.nn.Module
        if isinstance(activation, torch.nn.Module):
            act_module = activation
        elif isinstance(activation, str):
            act_module = {
                "": torch.nn.Identity,
                "identity": torch.nn.Identity,
                "none": torch.nn.Identity,
                "relu": torch.nn.ReLU,
                "sigmoid": torch.nn.Sigmoid,
                "tanh": torch.nn.Tanh,
            }[activation]()
        elif activation is None:
            act_module = torch.nn.Identity()
        else:
            raise ValueError(f"unknown value for activation {activation}.")

        assert n_inputs > 0 and n_outputs > 0
        linear_module: torch.nn.Module = torch.nn.Linear(
            n_inputs, n_outputs, bias=isinstance(norm_module, torch.nn.Identity)
        )
        if isinstance(act_module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            weight_init.c2_msra_fill(linear_module)
        else:
            weight_init.c2_xavier_fill(linear_module)

        self.linear = linear_module
        self.norm = norm_module
        self.activation = act_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        if self.shortcut == True:
            out += x
        return out

    @property
    def in_shape(self) -> ShapeSpec:
        return ShapeSpec(channels=1, height=1, width=1)

    @property
    def out_shape(self) -> ShapeSpec:
        return ShapeSpec(width=self.linear.out_features)


@BLOCK_LAYERS_REGISTRY.register()
class DoubleFullyConnectedBlock(BlockLayerBase):
    """
    A serial pair of two `FullyConnectedBlock`s A and B, each with the standard
    linear -> activation -> norm forward. In contrast to simply having two
    instances of `FullyConnectedBlock` in sequence, this class is useful
    because the `shortcut` acts a residual connection that "spans" both blocks;
    this behavior is designed to closely resemble ResNet and is not possible
    in a pure `torch.nn.Sequence` of two `FullyConnectedBlock`s.
    """

    def __init__(
        self,
        n_inputs: int,
        n_middle: int,
        n_outputs: int,
        norm_a: Optional[Union[str, torch.nn.Module]] = "",
        activation_a: Optional[Union[str, torch.nn.Module]] = "relu",
        norm_b: Optional[Union[str, torch.nn.Module]] = "",
        activation_b: Optional[Union[str, torch.nn.Module]] = "relu",
        shortcut: bool = False,
    ):
        super(BlockLayerBase, self).__init__()

        if shortcut == True and n_inputs != n_outputs:
            raise ValueError(f"can't use shortcut when {n_inputs=} != {n_outputs=}.")
        self.shortcut: bool = shortcut

        assert n_middle >= 0, f"n_middle must be non-negative; got {n_middle=}"
        if n_middle == 0:
            n_middle = n_inputs

        block_a: FullyConnectedBlock = FullyConnectedBlock(
            n_inputs=n_inputs,
            n_outputs=n_middle,
            norm=norm_a,
            activation=activation_a,
            shortcut=False,
        )
        block_b: FullyConnectedBlock = FullyConnectedBlock(
            n_inputs=n_middle,
            n_outputs=n_outputs,
            norm=norm_b,
            activation=activation_b,
            shortcut=False,
        )
        self.block_a: FullyConnectedBlock = block_a
        self.block_b: FullyConnectedBlock = block_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.block_a(x)
        out_b = self.block_b(out_a)
        if self.shortcut == True:
            out_b += x
        return out_b

    @property
    def in_shape(self) -> ShapeSpec:
        return self.block_a.in_shape

    @property
    def out_shape(self) -> ShapeSpec:
        return self.block_b.out_shape

from abc import abstractmethod

from detectron2.layers import FrozenBatchNorm2d
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.utils.registry import Registry
import torch

__all__ = ["BLOCK_LAYERS_REGISTRY", "BlockLayerBase"]


BLOCK_LAYERS_REGISTRY = Registry("BLOCK_LAYERS")
BLOCK_LAYERS_REGISTRY.__doc__ = """
Registry for block layers, which combine multiple related torch Modules into
one unit.

Registered object must return an instance of :class:`torch.nn.Module`.
"""


class BlockLayerBase(torch.nn.Module):
    @property
    @abstractmethod
    def in_shape(self) -> ShapeSpec:
        ...

    @property
    @abstractmethod
    def out_shape(self) -> ShapeSpec:
        ...

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

from detectron2.utils.registry import Registry

__all__ = ["BACKBONE_REGISTRY"]


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from input examples.

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of
:class:`detectron2.modeling.backbone.Backbone`.
"""

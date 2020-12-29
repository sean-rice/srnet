from typing import Any, Dict

from detectron2.config.config import CfgNode
from detectron2.layers.shape_spec import ShapeSpec

from ..common.fully_connected import FullyConnectedSequence
from .build import BACKBONE_REGISTRY

__all__ = ["FullyConnectedBackbone"]


@BACKBONE_REGISTRY.register()
class FullyConnectedBackbone(FullyConnectedSequence):
    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        input_shape: ShapeSpec,
        node_path: str = "MODEL.FULLY_CONNECTED_BACKBONE",
    ) -> Dict[str, Any]:
        return super().from_config(cfg, input_shape, node_path)

from typing import Any, Dict

from detectron2.config.config import CfgNode
from detectron2.layers.shape_spec import ShapeSpec

from ..common.block_sequence import BlockSequence
from .build import BACKBONE_REGISTRY

__all__ = ["BlockSequenceBackbone"]


@BACKBONE_REGISTRY.register()
class BlockSequenceBackbone(BlockSequence):
    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        input_shape: ShapeSpec,
        node_path: str = "MODEL.BLOCK_SEQUENCE_BACKBONE",
    ) -> Dict[str, Any]:
        return super().from_config(cfg, input_shape, node_path)

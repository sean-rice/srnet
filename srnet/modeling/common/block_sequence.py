import re
from typing import Any, Dict, List, Optional, Sequence

from detectron2.config import CfgNode, configurable
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone import Backbone
import torch
import torch.nn
import torch.nn.init

from srnet.layers.block_base import BLOCK_LAYERS_REGISTRY
from srnet.utils._utils import find_cfg_node

from .types import BlockSpecification

__all__ = ["BlockSequence"]


class BlockSequence(Backbone):
    @configurable
    def __init__(
        self,
        out_features: Sequence[str],
        block_specs: List[BlockSpecification],
        input_shape: Optional[ShapeSpec],
    ):
        super().__init__()

        if not self._check_block_sizes(block_specs):
            raise ValueError(
                f"found mismatch between block outputs/inputs in sequence."
            )

        # check out features; allow "flatten", "block_{#}", or "out"
        assert len(out_features) > 0
        ok_features = re.compile("(^flatten$|^block_\d+$|^out$)")
        for feature_name in out_features:
            if ok_features.match(feature_name) is None:
                raise ValueError(
                    f'bad out_feature "{feature_name}"; try "flatten", "block_#", or "out".'
                )
            split = feature_name.split("block_")
            if len(split) > 1 and (int(split[1]) not in range(0, len(block_specs))):
                raise ValueError(
                    f'bad out_feature "{feature_name}"; block number invalid.'
                )
        self.out_features: List[str] = list(set(out_features))

        blocks = self._build_blocks(block_specs)
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.blocks = torch.nn.ModuleDict(
            {f"block_{i}": block for i, block in enumerate(blocks)}
        )

    def _check_block_sizes(self, block_specs: Sequence[BlockSpecification]) -> bool:
        for (_, block1), (_, block2) in zip(block_specs, block_specs[1:]):
            try:
                n_outputs, n_inputs = block1["n_outputs"], block2["n_inputs"]
            except KeyError:
                return False
            if n_outputs != n_inputs:
                return False
        return True

    def _build_blocks(
        self, block_specs: List[BlockSpecification]
    ) -> List[torch.nn.Module]:
        blocks: List[torch.nn.Module] = []
        for block_class_name, block_args in block_specs:
            block_class = BLOCK_LAYERS_REGISTRY.get(block_class_name)
            block = block_class(**block_args)
            blocks.append(block)
        return blocks

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        x = self.flatten(input)
        for block_name, block in self.blocks.items():
            x = block(x)
            if block_name in self.out_features:
                result[block_name] = x
        if "out" in self.out_features:
            result["out"] = x
        return result

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        shapes: Dict[str, ShapeSpec] = {}
        for out_feature in self.out_features:
            if out_feature == "flatten":
                shapes[out_feature] = ShapeSpec(
                    width=self.blocks["block_0"].linear.in_features
                )
            elif out_feature == "out":
                shapes[out_feature] = list(self.blocks.values())[-1].out_shape
            else:
                shapes[out_feature] = self.blocks[out_feature].out_shape
        return shapes

    @classmethod
    def from_config(
        cls, cfg: CfgNode, input_shape: ShapeSpec, node_path: str
    ) -> Dict[str, Any]:
        node = find_cfg_node(cfg, node_path)
        return {
            "out_features": node.OUT_FEATURES,
            "block_specs": node.BLOCK_SPECIFICATIONS,
            "input_shape": input_shape,
        }

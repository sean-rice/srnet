from collections import OrderedDict
import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from detectron2.config import CfgNode, configurable
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone import Backbone
from fvcore.nn import weight_init
import torch
import torch.nn
import torch.nn.init

from srnet.utils._utils import find_cfg_node

__all__ = ["FullyConnectedSequence"]

_ACTIVATION_TO_MODULE: Dict[Optional[str], Type] = {
    None: torch.nn.Identity,
    "": torch.nn.Identity,
    "none": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}

_NORM_TO_MODULE: Dict[Optional[str], Type] = {
    None: torch.nn.Identity,
    "": torch.nn.Identity,
    "none": torch.nn.Identity,
    "bn": torch.nn.BatchNorm1d,
}


class FullyConnectedSequence(Backbone):
    """
    Deprecated; use BlockSequence.
    """

    @configurable
    def __init__(
        self,
        input_size: int,
        out_features: Sequence[str],
        layer_sizes: Sequence[int],
        layer_norms: Sequence[str],
        layer_activations: Sequence[str],
        input_shape: Optional[ShapeSpec],
    ):

        super().__init__()

        def _shape_defined_size(
            shape: Optional[ShapeSpec],
        ) -> Optional[Union[int, float]]:
            if shape is None:
                return None
            if shape.height is None or shape.width is None or shape.channels is None:
                return None
            return shape.height * shape.width * shape.channels

        # check input size
        if input_size % 1 != 0 or input_size < 0:
            raise ValueError(f"got non-integer or negative value {input_size=}")
        shape_size = _shape_defined_size(input_shape)
        if shape_size is not None and input_size != shape_size:
            raise ValueError(
                f"mismatch between input sizes {input_size=} and {shape_size=}"
            )

        # check layer sizes
        assert len(layer_sizes) > 0
        assert all(n % 1 == 0 if n is not None else False for n in layer_sizes)

        # check out features; allow "flatten", "block_{#}", or "out"
        assert len(out_features) > 0
        ok_features = re.compile("(^flatten$|^block_\d+$|^out$)")
        for feature_name in out_features:
            if ok_features.match(feature_name) is None:
                raise ValueError(
                    f'bad out_feature "{feature_name}"; try "flatten", "block_#", or "out".'
                )
            split = feature_name.split("block_")
            if len(split) > 1 and (int(split[1]) not in range(0, len(layer_sizes))):
                raise ValueError(
                    f'bad out_feature "{feature_name}"; block number invalid.'
                )
        self.out_features: List[str] = list(set(out_features))

        # norms
        assert len(layer_norms) == len(layer_sizes)
        layer_norms = [norm if norm is not None else "" for norm in layer_norms]
        assert all(norm in _NORM_TO_MODULE for norm in layer_norms)

        size_pairs: Sequence[Tuple[int, int]] = list(
            zip(itertools.chain([input_size], layer_sizes[:-1]), layer_sizes,)
        )

        # activations
        assert len(layer_activations) == len(layer_sizes)
        layer_activations = [a if a is not None else "" for a in layer_activations]
        assert all(a in _ACTIVATION_TO_MODULE for a in layer_activations)

        self.network = torch.nn.ModuleDict()
        self.network["flatten"] = torch.nn.Flatten()
        for i in range(0, len(size_pairs)):
            n_in_features, n_out_features = size_pairs[i]
            norm_class = _NORM_TO_MODULE[layer_norms[i]]
            act_class = _ACTIVATION_TO_MODULE[layer_activations[i]]
            block = self._build_block(
                n_in_features, n_out_features, norm_class, act_class,
            )
            self.network[f"block_{i}"] = block

    def _build_block(
        self,
        n_in_features: int,
        n_out_features: int,
        norm_class: Type,
        activation_class: Type,
    ) -> torch.nn.Module:
        # instantiate modules
        norm = norm_class(n_out_features)
        act = activation_class()
        linear = torch.nn.Linear(
            n_in_features, n_out_features, bias=isinstance(norm, torch.nn.Identity)
        )

        # initialize linear layer weights
        try:
            act_name: str = {
                torch.nn.Identity: "linear",
                torch.nn.ReLU: "relu",
                torch.nn.Tanh: "tanh",
                torch.nn.Sigmoid: "sigmoid",
            }[activation_class]
            gain = torch.nn.init.calculate_gain(act_name)
        except:
            act_name = "linear"
            gain = 1

        if act_name in ["relu", "leaky_relu"]:
            weight_init.c2_msra_fill(linear)
        else:
            weight_init.c2_xavier_fill(linear)

        block = torch.nn.Sequential(
            OrderedDict([("linear", linear), ("norm", norm), ("activation", act)])
        )
        return block

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        x = input
        for module_name, module in self.network.items():
            x = module(x)
            if module_name in self.out_features:
                result[module_name] = x
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
                    width=self.network["block_0"].linear.in_features
                )
            elif out_feature == "out":
                last_linear: torch.nn.Linear = (list(self.network.values())[-1]).linear
                shapes[out_feature] = ShapeSpec(width=last_linear.out_features)
            else:
                shapes[out_feature] = ShapeSpec(
                    width=self.network[out_feature].linear.out_features
                )
        return shapes

    @classmethod
    def from_config(
        cls, cfg: CfgNode, input_shape: ShapeSpec, node_path: str
    ) -> Dict[str, Any]:
        node = find_cfg_node(cfg, node_path)
        return {
            "input_size": node.INPUT_SIZE,
            "out_features": node.OUT_FEATURES,
            "layer_sizes": node.LAYER_SIZES,
            "layer_norms": node.LAYER_NORMS,
            "layer_activations": node.LAYER_ACTIVATIONS,
            "input_shape": input_shape,
        }

from typing import Any, Dict, Optional, Set, Tuple

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn
import torch.nn.functional
import torch.nn.init

from srnet.utils._utils import find_cfg_node

from ..common.types import Losses
from .classifier_head import CLASSIFIER_HEAD_REGISTRY, ClassifierHead

__all__ = ["LinearClassifierHead"]


@CLASSIFIER_HEAD_REGISTRY.register()
class LinearClassifierHead(ClassifierHead):
    """
    A classifier head that simply applies a linear layer to convert the input
    feature maps into a tensor of class activations.
    """

    @configurable
    def __init__(
        self,
        num_classes: int,
        in_feature: str,
        input_shape: Dict[str, ShapeSpec],
        loss_weight: float = ClassifierHead._loss_weight_default,
        loss_key: str = ClassifierHead._loss_key_default,
    ):
        super().__init__()
        self.in_feature: str = in_feature
        self.loss_weight: float = loss_weight
        self._loss_key: str = loss_key

        assert num_classes > 1
        assert (
            in_feature in input_shape
        ), f'provided input_shape doesn\'t have in_feature key "{in_feature}"'
        self.linear = torch.nn.Linear(
            input_shape[in_feature].width, num_classes
        )  # shape (N, W) -> (N, num_classes)
        torch.nn.init.xavier_normal_(self.linear.weight)

    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.CLASSIFIER_HEAD",
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        node = find_cfg_node(cfg, node_path)
        num_classes: int = node.NUM_CLASSES
        in_features: str = node.IN_FEATURES
        assert (
            len(in_features) == 1
        ), f"LinearClassifierHead supports only a single-item IN_FEATURES, got {in_features}"
        loss_weight: float = node.LOSS_WEIGHT
        loss_key: str = node.LOSS_KEY
        return {
            "num_classes": num_classes,
            "in_feature": in_features[0],
            "input_shape": input_shape,
            "loss_weight": loss_weight,
            "loss_key": loss_key,
        }

    @property
    def num_classes(self) -> int:
        return self.linear.out_features

    @property
    def loss_keys(self) -> Set[str]:
        return {self._loss_key}

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Losses]:
        cls_scores: torch.Tensor = self.linear(
            features[self.in_feature]
        )  # (N, self.num_classes)

        cls_losses: Losses = {}
        if self.training:
            assert targets is not None
            cls_loss = self.loss_weight * torch.nn.functional.cross_entropy(
                cls_scores, targets
            )
            cls_losses[self._loss_key] = cls_loss
        return cls_scores, cls_losses

from typing import Any, Dict, Optional, Set, Tuple

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn
import torch.nn.functional

from srnet.utils._utils import find_cfg_node

from ..common.types import Losses
from .classifier_head import CLASSIFIER_HEAD_REGISTRY, ClassifierHead

__all__ = ["PoolingClassifierHead"]


@CLASSIFIER_HEAD_REGISTRY.register()
class PoolingClassifierHead(ClassifierHead):
    """
    A classifier head that applies global pooling (avg or max) followed by a 
    linear layer to convert the input feature maps into a tensor of class
    activations.
    """

    @configurable
    def __init__(
        self,
        num_classes: int,
        pool_method: str,
        in_feature: str,
        input_shape: Dict[str, ShapeSpec],
        loss_weight: float = ClassifierHead._loss_weight_default,
        loss_key: str = ClassifierHead._loss_key_default,
    ):
        super().__init__()
        self.in_feature: str = in_feature
        self.loss_weight: float = loss_weight
        self._loss_key: str = loss_key

        pooler: torch.nn.Module
        if pool_method == "avg":
            pooler = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool_method == "max":
            pooler = torch.nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f'bad pool_method "{pool_method}"; try "avg" or "max".')
        self.pooler: torch.nn.Module = pooler  # shape (N, C, H, W) -> (N, C, 1, 1)

        assert num_classes > 1

        assert (
            in_feature in input_shape
        ), f'provided input_shape doesn\'t have in_feature key "{in_feature}"'
        self.fc = torch.nn.Linear(
            input_shape[in_feature].channels, num_classes
        )  # shape (N, C) -> (N, num_classes)

    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.CLASSIFIER_HEAD",
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_node = find_cfg_node(cfg, node_path)
        num_classes: int = target_node.NUM_CLASSES
        pool_method: str = target_node.POOL_METHOD
        in_features: str = target_node.IN_FEATURES
        assert (
            len(in_features) == 1
        ), f"PoolingClassifierHead supports only a single-item IN_FEATURES, got {in_features}"
        loss_weight: float = target_node.LOSS_WEIGHT
        loss_key: str = target_node.LOSS_KEY
        return {
            "num_classes": num_classes,
            "pool_method": pool_method,
            "in_feature": in_features[0],
            "input_shape": input_shape,
            "loss_weight": loss_weight,
            "loss_key": loss_key,
        }

    @property
    def num_classes(self) -> int:
        return self.fc.out_features

    @property
    def loss_keys(self) -> Set[str]:
        return {self._loss_key}

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Losses]:
        pooled: torch.Tensor = self.pooler(features[self.in_feature])  # (N, C, 1, 1)
        pooled = pooled.squeeze()  # ([N?,] C)
        # if N == 1, it will get squeezed out; check for that (len(shape) == 1)
        # and re-add the batch dimension if it happens.
        if len(pooled.shape) == 1:
            pooled = pooled.unsqueeze(dim=0)  # (1, C)
        cls_scores: torch.Tensor = self.fc(pooled)  # (N, self.num_classes)

        cls_losses: Losses = {}
        if self.training:
            assert targets is not None
            cls_loss = self.loss_weight * torch.nn.functional.cross_entropy(
                cls_scores, targets
            )
            cls_losses[self._loss_key] = cls_loss
        return cls_scores, cls_losses

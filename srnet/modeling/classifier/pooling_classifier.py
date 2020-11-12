from typing import Any, Dict

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn

from .classifier_head import CLASSIFIER_HEAD_REGISTRY, ClassifierHead


@CLASSIFIER_HEAD_REGISTRY.register()
class PoolingClassifierHead(ClassifierHead):
    @configurable
    def __init__(
        self, pool_method: str, num_classes: int, input_shape: ShapeSpec,
    ):
        pooler: torch.nn.Module
        if pool_method == "avg":
            pooler = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool_method == "max":
            pooler = torch.nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f'bad pool_method "{pool_method}"; try "avg" or "max".')
        self.pooler: torch.nn.Module = pooler  # shape (N, C, H, W) -> (N, C)

        assert num_classes > 0

        self.fc = torch.nn.Linear(
            input_shape.channels, num_classes
        )  # shape (N, C) -> (N, num_classes)

    @classmethod
    def from_config(cls, cfg: CfgNode, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        pool_method: str = cfg.MODEL.POOLING_CLASSIFIER_HEAD.POOL_METHOD
        num_classes: int = cfg.MODEL.CLASSIFIER.NUM_CLASSES
        in_feature: str = cfg.MODEL.CLASSIFIER.IN_FEATURES
        assert isinstance(in_feature, str)
        assert "input_shape" in kwargs
        assert in_feature in kwargs["input_shape"]
        input_shape: ShapeSpec = kwargs["input_shape"][in_feature]
        return {
            "pool_method": pool_method,
            "num_classes": num_classes,
            "input_shape": input_shape,
        }

    @property
    def num_classes(self) -> int:
        return self.fc.out_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled: torch.Tensor = self.pooler(features)
        cls_scores: torch.Tensor = self.fc(pooled)
        return cls_scores

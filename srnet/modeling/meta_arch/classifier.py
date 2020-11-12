from typing import Optional, Tuple

from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone
import torch

from ..classifier.classifier_head import ClassifierHead
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Classifier(torch.nn.Module):
    """
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        classifier_head: ClassifierHead,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
    ):
        super().__init__()
        self.backbone: Backbone = backbone
        self.classifier_head: ClassifierHead = classifier_head

        self.input_format: Optional[str] = input_format

        self.pixel_mean: torch.Tensor
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.pixel_std: torch.Tensor
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))  # type: ignore[call-arg]

    def forward(self):
        pass

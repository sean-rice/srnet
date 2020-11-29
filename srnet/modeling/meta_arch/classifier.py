from typing import Any, Dict, List, Optional, Tuple, Union

from detectron2.config import CfgNode, configurable
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.structures import ImageList
import torch

from ..classifier.classifier_head import ClassifierHead, build_classifier_head
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Classifier(torch.nn.Module):
    """
    A classifier network, composed of a backbone and a classifier head.
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

        assert backbone is not None
        assert classifier_head is not None

        self.backbone: Backbone = backbone
        self.classifier_head: ClassifierHead = classifier_head

        self.input_format: Optional[str] = input_format

        self.pixel_mean: torch.Tensor
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.pixel_std: torch.Tensor
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))  # type: ignore[call-arg]

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Dict[str, Any]:
        backbone: Backbone = build_backbone(cfg)
        out_shape: Dict[str, ShapeSpec] = backbone.output_shape()
        classifier_head = build_classifier_head(cfg, out_shape)
        return {
            "backbone": backbone,
            "classifier_head": classifier_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(
        self, batched_inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> Union[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Args:
            batched_inputs (List[Dict[str, Any]]): a list, batched outputs of
                :class:`DatasetMapper`. Each item in the list contains the
                inputs for one image. For now, each item in the list is a dict
                that contains:
                * image: Tensor, image in (C, H, W) format.
                * class_label: ground-truth class, as single-item tensor
            normalize (bool): whether or not to normalize the incoming images.
        
        Returns:
            results (List[Dict[str, Any]] or Dict[str, torch.Tensor]):
                During training, returns a dictionary of losses.

                During inference, a list of dicts, where each dict is the
                output for one input image.
                The dict contains: 1) a key "pred_class_scores" whose value is
                a :class:`torch.Tensor` with length equal to the number of
                classes.
        """
        if not self.training:
            return self.inference(batched_inputs, normalize)

        images = self.preprocess_image(batched_inputs)
        targets = self.preprocess_target(batched_inputs)
        features: Dict[str, torch.Tensor] = self.backbone(images.tensor)
        classifier_losses: Dict[str, torch.Tensor]
        _, classifier_losses = self.classifier_head(features, targets=targets)
        return classifier_losses

    def inference(
        self, batched_inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> List[Dict[str, Any]]:
        images = self.preprocess_image(batched_inputs)
        features: Dict[str, torch.Tensor] = self.backbone(images.tensor)
        class_scores: torch.Tensor
        class_scores, _ = self.classifier_head(features)
        return [{"pred_class_scores": scores} for scores in class_scores]

    def preprocess_image(
        self, batched_inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> ImageList:
        """
        Preprocess the input images, including normalization, padding, and
        batching.
        """
        images: List[torch.Tensor] = [
            x["image"].to(self.device) for x in batched_inputs
        ]
        if normalize == True:
            images = [self._normalize(x) for x in images]
        image_list = ImageList.from_tensors(images)
        return image_list

    def preprocess_target(self, batched_inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Preprocess the input items into a tensor of target classes.
        """
        targets: torch.Tensor = torch.stack(
            [item["class_label"] for item in batched_inputs]
        ).to(device=self.device, dtype=torch.long)
        return targets

    def _normalize(
        self,
        image: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean = self.pixel_mean if mean is None else mean
        std = self.pixel_std if std is None else std
        return (image - mean) / std

    def _denormalize(
        self,
        image: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean = self.pixel_mean if mean is None else mean
        std = self.pixel_std if std is None else std
        return (image * std) + mean

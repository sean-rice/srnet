from typing import Any, Dict, List, Optional, Sequence, Tuple

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn

from srnet.modeling.common.fully_connected import FullyConnectedSequence
from srnet.utils._utils import find_cfg_node

from ..common.types import Losses
from .unsupervised_head import (
    UNSUPERVISED_HEAD_REGISTRY,
    UnsupervisedHead,
    UnsupervisedOutput,
)

__all__ = ["FullyConnectedImageDecoder"]


@UNSUPERVISED_HEAD_REGISTRY.register()
class FullyConnectedImageDecoder(UnsupervisedHead):
    """
    An image decoder that takes a feature map as input and uses a series of
    fully-connected layers to decode an image.

    Note that both the input features and the output image sizes are fixed and
    must be pre-specified.
    """

    @configurable
    def __init__(
        self,
        in_feature: str,
        input_shape: Dict[str, ShapeSpec],
        layer_sizes: Sequence[int],
        layer_norms: Sequence[str],
        layer_activations: Sequence[str],
        output_shape: Sequence[int],
        loss_weight: float,
        loss_key: str = "loss_image_decoder",
    ):
        super(UnsupervisedHead, self).__init__()

        self.in_feature: str = in_feature
        self.output_size: torch.Size = torch.Size(output_shape)
        self.loss_weight: float = loss_weight
        self.loss_key: str = loss_key

        self.network = FullyConnectedSequence(
            input_shape[in_feature].width,
            ("out",),
            layer_sizes,
            layer_norms,
            layer_activations,
            None,
        )
        self.reshape = torch.nn.Unflatten(1, self.output_size)

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]]
    ) -> Tuple[UnsupervisedOutput, Losses]:
        """
        Decode a batch of feature embeddings back into images.

        Args:
            features (dict): A dict from a feature's name ("out", etc.) to the
                feature itself (a `torch.Tensor`).
            targets (dict): A dict containing the target images to decode to.
                This should contain an "images" key with an value of type
                `ImageList`.
            
        Returns:
            results (dict): A dict containing a single key, `"decoded_images"`,
                with a value of type `torch.Tensor` containing the decoded
                images in standard torch.Size([N,C,H,W]) format.
            losses (dict): In inference, an empty dict. In training, a dict
                containing a single key `self.LOSS_KEY` containing the MSE
                loss between the actual input images and the decoded
                reconstructions.
        """
        decoded_images = self.layers(features)

        results: UnsupervisedOutput = {"decoded_images": decoded_images}
        losses: Losses = {}
        if self.training:
            assert targets is not None
            original_images: torch.Tensor = targets[
                "images"
            ].tensor  # ImageList -> Tensor
            losses[self.loss_key] = (
                torch.nn.functional.mse_loss(
                    decoded_images, original_images, reduction="mean"
                )
                * self.loss_weight
            )
        return results, losses

    def layers(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply the module's layers (the fully connected network, then the
        unflatten/reshaping) to feature maps to obtain decoded images.

        Args:
            features (dict): see `self.forward()`

        Returns:
            decoded_images (torch.Tensor): A tensor containing the decoded
                images in standard torch.Size([N,C,H,W]) format.
        """
        features: torch.Tensor = features[self.in_feature]  # type: ignore[no-redef]
        decoded_images = self.network(features)["out"]
        decoded_images = self.reshape(decoded_images)
        return decoded_images

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.IMAGE_DECODER",
    ) -> Dict[str, Any]:
        target_node = find_cfg_node(cfg, node_path)
        assert len(target_node.IN_FEATURES) == 1

        return {
            "in_feature": target_node.IN_FEATURES[0],
            "input_shape": input_shape,
            "layer_sizes": target_node.LAYER_SIZES,
            "layer_norms": target_node.LAYER_NORMS,
            "layer_activations": target_node.LAYER_ACTIVATIONS,
            "output_shape": (
                len(cfg.MODEL.PIXEL_MEAN),
                target_node.OUTPUT_HEIGHT,
                target_node.OUTPUT_WIDTH,
            ),
            "loss_weight": target_node.LOSS_WEIGHT,
            "loss_key": target_node.LOSS_KEY,
        }

    @classmethod
    def into_per_item_iterable(
        cls, network_output: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        return [img for img in network_output["decoded_images"]]

    @classmethod
    def postprocess(  # type: ignore[override]
        cls,
        result: torch.Tensor,
        img_size: Tuple[int, int],
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """
        Return image decoder predictions in the original resolution.

        Args:
            result (Tensor): Image decoder output prediction. A tensor of
                shape (C, H, W), where C is the number of channels (likely 3),
                and H, W are the height and width of the prediction.
            img_size (tuple[int, int]): image size that the image decoder is
                taking as input.
            output_height, output_width: the desired output resolution.

        Returns:
            image decoder prediction (Tensor): A tensor of the shape
                (C, output_height, output_width) that contains the prediction.
        """
        result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
        result = torch.nn.functional.interpolate(
            result,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        return result

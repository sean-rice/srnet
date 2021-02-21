from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch
import torch.nn

from srnet.utils._utils import find_cfg_node

from ..common.block_sequence import BlockSequence
from ..common.types import BlockSpecification, Losses
from .unsupervised_head import (
    UNSUPERVISED_HEAD_REGISTRY,
    UnsupervisedHead,
    UnsupervisedOutput,
)

__all__ = ["BlockSequenceImageDecoder"]


@UNSUPERVISED_HEAD_REGISTRY.register()
class BlockSequenceImageDecoder(UnsupervisedHead):
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
        block_specs: Sequence[BlockSpecification],
        output_shape: Sequence[int],
        loss_weight: float,
        loss_key: str = "loss_image_decoder",
        output_key: str = "decoded_image",
    ):
        super(UnsupervisedHead, self).__init__()

        self.in_feature: str = in_feature
        self.output_size: torch.Size = torch.Size(output_shape)
        self.loss_weight: float = loss_weight
        self.loss_key: str = loss_key
        self.output_key: str = output_key

        self.blocks: BlockSequence = BlockSequence(
            out_features=("out",), block_specs=block_specs, input_shape=None
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
            results (dict): A dict containing a single key, `self.output_key`,
                with a value of type `torch.Tensor` containing the decoded
                images in standard torch.Size([N,C,H,W]) format.
            losses (dict): In inference, an empty dict. In training, a dict
                containing a single key `self.LOSS_KEY` containing the MSE
                loss between the actual input images and the decoded
                reconstructions.
        """
        decoded_images = self.layers(features)

        results: UnsupervisedOutput = {self.output_key: decoded_images}
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
        decoded_images = self.blocks(features)["out"]
        decoded_images = self.reshape(decoded_images)
        return decoded_images

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.IMAGE_DECODER",
    ) -> Dict[str, Any]:
        node = find_cfg_node(cfg, node_path)
        assert len(node.IN_FEATURES) == 1
        return {
            "in_feature": node.IN_FEATURES[0],
            "block_specs": node.BLOCK_SPECIFICATIONS,
            "output_shape": (
                len(cfg.MODEL.PIXEL_MEAN),
                node.OUTPUT_HEIGHT,
                node.OUTPUT_WIDTH,
            ),
            "loss_weight": node.LOSS_WEIGHT,
            "loss_key": node.LOSS_KEY,
            "output_key": node.OUTPUT_KEY,
        }

    def into_per_item_iterable(
        self, network_output: UnsupervisedOutput
    ) -> List[Dict[str, torch.Tensor]]:
        return [{self.output_key: img} for img in network_output[self.output_key]]

    def postprocess(  # type: ignore[override]
        self,
        result: Mapping[str, Optional[torch.Tensor]],
        *,
        img_size: Tuple[int, int],
        output_height: int,
        output_width: int,
        **kwargs: Any,
    ) -> Dict[str, Optional[torch.Tensor]]:
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
        r: Optional[torch.Tensor] = result.get(self.output_key, None)  # image
        if r is None:
            return {self.output_key: None}
        r = r[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
        r = torch.nn.functional.interpolate(
            r, size=(output_height, output_width), mode="bilinear", align_corners=False,
        )[0]
        return {self.output_key: r}

    @property
    def loss_keys(self) -> Set[str]:
        return {self.loss_key}

    @property
    def output_keys(self) -> Set[str]:
        return {self.output_key}

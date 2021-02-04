import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from detectron2.config import CfgNode, configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch

from srnet.utils._utils import find_cfg_node

from ..common.types import Losses
from .unsupervised_head import (
    UNSUPERVISED_HEAD_REGISTRY,
    UnsupervisedHead,
    UnsupervisedOutput,
)

__all__ = ["ImageDecoder"]


@UNSUPERVISED_HEAD_REGISTRY.register()
class ImageDecoder(UnsupervisedHead):
    @configurable
    def __init__(
        self,
        in_features: Iterable[str],
        input_shape: Dict[str, ShapeSpec],
        output_channels: int,
        common_stride: int,
        scale_heads_dim: int,
        scale_heads_norm: str,
        predictor_depth: int,
        predictor_dim: int,
        predictor_norm: str,
        loss_weight: float,
        loss_key: str = "loss_image_decoder",
        output_key: str = "decoded_images",
    ):
        super(UnsupervisedHead, self).__init__()

        self.in_features: List[str] = list(copy.copy(in_features))
        self.output_channels: int = output_channels
        self.common_stride: int = common_stride
        self.loss_weight: float = loss_weight
        self.loss_key: str = loss_key
        self.output_key: str = output_key

        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}

        # build scale heads
        self.scale_heads: List[torch.nn.Module] = []
        for in_feature in self.in_features:
            head_ops: List[torch.nn.Module] = []
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for k in range(head_length):
                norm_module: Optional[torch.nn.Module]
                norm_module = get_norm(scale_heads_norm, scale_heads_dim)
                conv = Conv2d(
                    scale_heads_dim if k > 0 else feature_channels[in_feature],
                    scale_heads_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm_module is None),
                    norm=norm_module,
                    activation=torch.nn.functional.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        torch.nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.scale_heads.append(torch.nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        # build pixel predictor
        predictor_ops: List[torch.nn.Module] = []
        for i in range(predictor_depth - 1):
            norm_module = get_norm(predictor_norm, predictor_dim)
            pred_module = Conv2d(
                predictor_dim if i > 0 else scale_heads_dim,
                predictor_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=(norm_module is None),
                norm=norm_module,
                activation=torch.nn.functional.relu,
            )
            weight_init.c2_msra_fill(pred_module)
            predictor_ops.append(pred_module)
        penultimate_dim: int = predictor_dim if predictor_depth > 1 else scale_heads_dim
        predictor_ops.append(
            Conv2d(
                penultimate_dim,
                self.output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        assert (
            len(predictor_ops) == predictor_depth
        ), f"{len(predictor_ops)} != {predictor_depth}"
        weight_init.c2_msra_fill(predictor_ops[-1])
        self.predictor = torch.nn.Sequential(*predictor_ops)

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]]
    ) -> Tuple[UnsupervisedOutput, Losses]:
        """
        Decode a batch of feature embeddings back into images.

        Args:
            features (dict): A dict from a feature's name ("p2", etc.) to the
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
        Apply the module's layers (each of the scale heads, then pixel
        predictor) to feature maps to obtain decoded images.

        Args:
            features (dict): see `self.forward()`

        Returns:
            decoded_images (torch.Tensor): A tensor containing the decoded
                images in standard torch.Size([N,C,H,W]) format.
        """
        x: torch.Tensor
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        decoded_images = torch.nn.functional.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        return decoded_images

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.IMAGE_DECODER",
    ) -> Dict[str, Any]:
        target_node = find_cfg_node(cfg, node_path)

        return {
            "in_features": target_node.IN_FEATURES,
            "input_shape": input_shape,
            "output_channels": len(cfg.MODEL.PIXEL_MEAN),
            "common_stride": target_node.COMMON_STRIDE,
            "scale_heads_dim": target_node.SCALE_HEADS_DIM,
            "scale_heads_norm": target_node.SCALE_HEADS_NORM,
            "predictor_depth": target_node.PREDICTOR_DEPTH,
            "predictor_dim": target_node.PREDICTOR_DIM,
            "predictor_norm": target_node.PREDICTOR_NORM,
            "loss_weight": target_node.LOSS_WEIGHT,
            "loss_key": target_node.LOSS_KEY,
            "output_key": target_node.OUTPUT_KEY,
        }

    def into_per_item_iterable(
        self, network_output: UnsupervisedOutput
    ) -> List[torch.Tensor]:
        return [img for img in network_output[self.output_key]]

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

    @property
    def loss_keys(self) -> Set[str]:
        return {self.loss_key}

    @property
    def output_keys(self) -> Set[str]:
        return {self.output_key}

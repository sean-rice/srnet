from typing import Any, Dict, Iterable, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
import numpy as np
import torch

from ..unsupervised.unsupervised_head import UnsupervisedHead, build_unsupervised_head

__all__ = ["UxRCNN"]


class UxRCNN(torch.nn.Module):
    """
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: torch.nn.Module,
        roi_heads: torch.nn.Module,
        unsupervised_head: UnsupervisedHead,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__()
        self.backbone: Backbone = backbone
        self.proposal_generator: torch.nn.Module = proposal_generator
        self.roi_heads: torch.nn.Module = roi_heads
        self.unsupervised_head: UnsupervisedHead = unsupervised_head

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.pixel_mean: torch.Tensor
        self.pixel_std: torch.Tensor
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))  # type: ignore[call-arg]
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))  # type: ignore[call-arg]
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        out_shape = backbone.output_shape()
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, out_shape),
            "roi_heads": build_roi_heads(cfg, out_shape),
            "unsupervised_head": build_unsupervised_head(cfg, out_shape),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, normalize: bool = True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed
                proposals.
                Other information that's included in the original dicts, such
                as:
                * "height", "width" (int): the output resolution of the model,
                  used in inference. See :meth:`postprocess` for details.
            normalize (bool): whether or not to normalize the incoming images.
                disabling normalization can be helpful when e.g. recursively
                and repeatedly feeding an autoencoder's output to the network
                as input.
        
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a
                :class:`Instances`, as well as an "unsupervised_output" key of
                unspecified (but likely :class:`torch.Tensor`) type.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks",
                "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        (
            batched_inputs,
            n_unlabeled,
            n_labeled,
            sorted_indices,
        ) = self.preprocess_batch_order(batched_inputs)
        i_labeled_0 = n_unlabeled  # *index* of first labeled is unlabeled count

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # run detection pipeline conditionally if any labeled examples exist
        gt_instances: Optional[List[Instances]]
        features_for_detector: Optional[Dict[str, torch.Tensor]]
        proposals: Optional[Any]
        proposal_losses: Dict[str, torch.Tensor]
        detector_losses: Dict[str, torch.Tensor]
        if n_labeled > 0:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs[i_labeled_0:]
            ]
            # split out labeled features from backbone result tensors
            features_for_detector = {}
            for f in features.keys():
                _, labeled = torch.split(features[f], (n_unlabeled, n_labeled), dim=0)
                features_for_detector[f] = labeled

            # generate or fetch proposals
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(
                    images, features_for_detector, gt_instances
                )
            else:
                assert "proposals" in batched_inputs[i_labeled_0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            # run roi heads (detector)
            _, detector_losses = self.roi_heads(
                images, features_for_detector, proposals, gt_instances
            )
        else:
            gt_instances = None
            features_for_detector = None
            proposals = None
            proposal_losses = {}
            detector_losses = {}

        # unsupervised objective head
        _, unsupervised_losses = self.unsupervised_head(
            features, {"images": images}
        )  # TODO

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def preprocess_image(
        self, batched_inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> "ImageList":
        r"""
        Preprocess the input images, including normalization, padding, and
        batching.
        """
        images: List[torch.Tensor] = [
            x["image"].to(self.device) for x in batched_inputs
        ]
        if normalize == True:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        image_list = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return image_list

    def preprocess_batch_order(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int, int, Iterable[int]]:
        r"""
        Preprocesses an input batch's list order such that the unlabeled
        examples-- those without an "instances" entry-- come before the labeled
        ones. This is important once the batch becomes a single `Tensor` and
        the items within the batch need to be routed to different components
        (such as the RPN and RoI heads) without breaking up the original
        :class:`Tensor` or making a copy.

        Args:
            batched_inputs (list): A list of dicts from a
                :class:`DatasetMapper` for the model to use as input, each item
                of which may or may not contain an "instances" entry.
        
        Returns:
            batched_inputs (list): The re-ordered inputs. This is a new list
                containing references to the original items in the input list.
            n_unlabeled (int): The number of unlabeled items from the input
                list.
            n_labeled (int): The number of labeled items from the input list.
            sorted_indices (Iterable[int]): The original index in the input of
                each item in the returned list. This allows for undoing the
                sort (putting items back in their original order) at a later
                stage.
        """
        with torch.no_grad():
            has_instances = torch.tensor(
                [(1 if "instances" in inp else 0) for inp in batched_inputs],
                dtype=torch.int,
                device=torch.device("cpu"),
            )
            sorted_indices: Iterable[int]
            sorted_indices = (int(i.item()) for i in torch.argsort(has_instances))
            n_labeled = int(torch.sum(has_instances).item())
            n_unlabeled = len(batched_inputs) - n_labeled
            batched_inputs = [batched_inputs[i] for i in sorted_indices]
        return batched_inputs, n_unlabeled, n_labeled, sorted_indices

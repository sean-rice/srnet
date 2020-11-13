from typing import Any, Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
import torch

from ..unsupervised.unsupervised_head import UnsupervisedHead, build_unsupervised_head
from .build import META_ARCH_REGISTRY

__all__ = ["UxRCNN"]


@META_ARCH_REGISTRY.register()
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
            results (List[Dict[str, Any]] or Dict[str, torch.Tensor]):
                During training, returns a dictionary of losses.

                During inference, a list of dicts, where each dict is the
                output for one input image.
                The dict contains: 1) a key "instances" whose value is a
                :class:`Instances`, which has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks",
                "pred_keypoints"; and 2) an "unsupervised" key of unspecified
                (but likely :class:`torch.Tensor`-or-`None`) type.
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
        proposal_losses: Dict[str, torch.Tensor] = {}
        detector_losses: Dict[str, torch.Tensor] = {}
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

            # run roi heads (detector)
            _, detector_losses = self.roi_heads(
                images, features_for_detector, proposals, gt_instances
            )
        else:
            gt_instances = None
            features_for_detector = None
            proposals = None

        # unsupervised objective head
        unsupervised_losses: Dict[str, torch.Tensor]
        _, unsupervised_losses = self.unsupervised_head(
            features, {"inputs": batched_inputs, "images": images}
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(unsupervised_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, Any]],
        do_unsupervised: bool = True,
        do_postprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform inference on a batch of images.

        NOTE: To accomodate torch.nn.Module hooks, etc, it is probably better
        to call the module itself while `self.training == False` than to
        directly call this method.

        Args:
            batched_inputs (List[Dict[str, Any]]): see :meth:`self.forward`.
            do_unsupervised (bool): whether or not to run the unsupervised
                objective head.
            do_postprocess (bool): whether or not to post-process the results
                of both the roi heads and the unsupervised objective.
        
        Returns:
            results (List[Dict[str, Any]]): see :meth:`self.forward`.
        """
        images = self.preprocess_image(batched_inputs)
        features: Dict[str, torch.Tensor] = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        detector_results: Optional[List[Instances]]
        detector_results, _ = self.roi_heads(images, features, proposals, None)

        unsupervised_results: Optional[List[torch.Tensor]] = None
        if do_unsupervised == True and isinstance(
            self.unsupervised_head, torch.nn.Module
        ):
            unsup_results_raw, _ = self.unsupervised_head(
                features, {"inputs": batched_inputs, "images": images}
            )
            unsupervised_results = self.unsupervised_head.into_per_item_iterable(
                unsup_results_raw
            )

        if do_postprocess == True:
            if detector_results is None or unsupervised_results is None:
                none_list = [None] * len(batched_inputs)
            else:
                none_list = None  # type: ignore
            results = self._postprocess(
                batched_inputs,
                images.image_sizes,
                none_list if detector_results is None else detector_results,
                none_list if unsupervised_results is None else unsupervised_results,  # type: ignore[arg-type]
            )
        else:
            raise NotImplementedError("not implemeneted; enable post-processing.")
            # results: List[Dict[str, Any]] = [{} for _ in range(len(batched_inputs))]
            # for i, result in enumerate(results):
            #    result["instances"] = #TODO

        return results

    def _postprocess(
        self,
        batched_inputs: List[Dict[str, Any]],
        images_sizes: List[Tuple[int, int]],
        detector_results: List[Optional[Instances]],
        unsupervised_results: List[Optional[torch.Tensor]],
    ) -> List[Dict[str, Any]]:
        batched_inputs
        n_inputs = len(batched_inputs)
        if n_inputs != len(images_sizes):
            raise ValueError(f"length mismatch; {n_inputs=} but {len(images_sizes)=}")
        if n_inputs != len(detector_results):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(detector_results)=}"
            )
        if n_inputs != len(unsupervised_results):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(unsupervised_results)=}"
            )

        results: List[Dict[str, Any]] = [{} for _ in range(n_inputs)]
        for i in range(len(batched_inputs)):
            image_input = batched_inputs[i]
            image_size = images_sizes[i]
            image_instances = detector_results[i]
            image_unsup_result = unsupervised_results[i]

            h: int = image_input.get("height", image_size[0])
            w: int = image_input.get("width", image_size[1])
            if image_instances is not None:
                r = detector_postprocess(image_instances, h, w)
                results[i]["instances"] = r
            if image_unsup_result is not None:
                u = self.unsupervised_head.postprocess(
                    image_unsup_result, image_size, h, w
                )
                results[i]["unsupervised"] = u
        return results

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
        image_list = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return image_list

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

    def preprocess_batch_order(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int, int, List[int]]:
        """
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
            sorted_indices (List[int]): The original index in the input of
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
            sorted_indices = [int(i.item()) for i in torch.argsort(has_instances)]
            n_labeled = int(torch.sum(has_instances).item())
            n_unlabeled = len(batched_inputs) - n_labeled
            batched_inputs = [batched_inputs[i] for i in sorted_indices]
        return batched_inputs, n_unlabeled, n_labeled, sorted_indices

    def visualize_training(self, *args, **kwargs):
        pass

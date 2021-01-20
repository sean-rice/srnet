from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from detectron2.config import CfgNode, configurable
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone import Backbone
import torch

from ..classifier.classifier_head import ClassifierHead
from ..common.types import Losses
from ..unsupervised.unsupervised_head import (
    UnsupervisedHead,
    UnsupervisedOutput,
    build_unsupervised_head,
)
from ..unsupervised.utils import preprocess_batch_order
from .build import META_ARCH_REGISTRY
from .classifier import Classifier, ClassifierResult


@META_ARCH_REGISTRY.register()
class UxClassifier(Classifier):
    """
    An Unsupervised eXtended classifier network, composed of a backbone, a
    supervised classifier head, and an unsupervised objective head.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        classifier_head: ClassifierHead,
        unsupervised_head: UnsupervisedHead,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__(backbone, classifier_head, pixel_mean, pixel_std, input_format)

        assert unsupervised_head is not None
        self.unsupervised_head: UnsupervisedHead = unsupervised_head

        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Dict[str, Any]:
        args: Dict[str, Any] = super().from_config(cfg)
        out_shape: Dict[str, ShapeSpec] = args["backbone"].output_shape()
        unsupervised_head = build_unsupervised_head(cfg, out_shape)
        args["unsupervised_head"] = unsupervised_head
        args["vis_period"] = cfg.VIS_PERIOD
        return args

    def forward(
        self, batched_inputs: List[Dict[str, Any]], normalize: bool = True
    ) -> Union[List[Dict[str, Any]], Losses]:
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
                classes; 2) an "unsupervised" key of unspecified (but likely
                :class:`torch.Tensor`-or-`None`) type.
        """
        if not self.training:
            return self.inference(batched_inputs, normalize)

        images = self.preprocess_image(batched_inputs, normalize)
        targets = self.preprocess_target(batched_inputs)

        results: ClassifierResult = self.layers(
            images, targets, unsup_targets={"inputs": batched_inputs, "images": images}
        )
        return results.losses

    def layers(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        unsup_targets: Optional[Dict[str, Any]] = None,
    ) -> ClassifierResult:
        features: Dict[str, torch.Tensor] = self.backbone(images)
        class_scores: torch.Tensor
        classifier_losses: Losses
        class_scores, classifier_losses = self.classifier_head(
            features, targets=targets
        )
        unsupervised_output: UnsupervisedOutput
        unsupervised_losses: Losses
        unsupervised_output, unsupervised_losses = self.unsupervised_head(
            features, unsup_targets
        )

        losses: Losses = {}
        losses.update(classifier_losses)
        losses.update(unsupervised_losses)
        return ClassifierResult(
            class_scores, losses, {"unsupervised_output": unsupervised_output}
        )

    def inference(
        self, batched_inputs: List[Dict[str, Any]], do_postprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        images = self.preprocess_image(batched_inputs)
        classifier_results: ClassifierResult = self.layers(
            images, targets=None, unsup_targets=None
        )

        if do_postprocess == True:
            results = self._postprocess(
                batched_inputs,
                images.image_sizes,
                cast(Sequence[torch.Tensor], classifier_results.class_scores),
                classifier_results.extras["unsupervised_output"],
            )
            return results
        else:
            raise NotImplementedError("not implemeneted; enable post-processing.")

    def _postprocess(
        self,
        batched_inputs: Sequence[Dict[str, Any]],
        images_sizes: Sequence[Tuple[int, int]],
        classifier_results: Optional[Sequence[Optional[torch.Tensor]]],
        unsupervised_output: Optional[UnsupervisedOutput],
    ) -> List[Dict[str, Any]]:

        n_inputs = len(batched_inputs)

        unsupervised_items: Optional[Sequence[Optional[torch.Tensor]]] = None
        if unsupervised_output is not None:
            unsupervised_items = self.unsupervised_head.into_per_item_iterable(
                unsupervised_output
            )

        if classifier_results is None or unsupervised_items is None:
            nones = (None,) * n_inputs
            if classifier_results is None:
                classifier_results = nones
            if unsupervised_items is None:
                unsupervised_items = nones

        if n_inputs != len(images_sizes):
            raise ValueError(f"length mismatch; {n_inputs=} but {len(images_sizes)=}")
        if n_inputs != len(classifier_results):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(classifier_results)=}"
            )
        if n_inputs != len(unsupervised_items):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(unsupervised_items)=}"
            )

        results: List[Dict[str, Any]] = [{} for _ in range(n_inputs)]
        for i in range(len(batched_inputs)):
            image_input = batched_inputs[i]
            image_size = images_sizes[i]
            image_class_scores = classifier_results[i]
            image_unsup_result = unsupervised_items[i]

            h: int = image_input.get("height", image_size[0])
            w: int = image_input.get("width", image_size[1])
            if image_class_scores is not None:
                results[i]["pred_class_scores"] = image_class_scores
            if image_unsup_result is not None:
                u = self.unsupervised_head.postprocess(
                    image_unsup_result, image_size, h, w
                )
                results[i]["unsupervised"] = u
        return results

    @classmethod
    def preprocess_batch_order(
        cls, batched_inputs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int, int, List[int]]:
        """
        See `srnet.modeling.unsupervised.utils.preprocess_batch_order`.

        Args:
            batched_inputs (list): A list of dicts from a
                :class:`DatasetMapper` for the model to use as input, each item
                of which may or may not contain a "class_label" entry.
        
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
        return preprocess_batch_order(
            batched_inputs, lambda example: 1 if "class_label" in example else 0
        )

    def visualize_training(self, *args, **kwargs):
        pass

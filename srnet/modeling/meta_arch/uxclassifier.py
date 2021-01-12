from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from detectron2.config import CfgNode, configurable
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.structures import ImageList
import torch

from ..classifier.classifier_head import ClassifierHead, build_classifier_head
from ..unsupervised.unsupervised_head import UnsupervisedHead, build_unsupervised_head
from ..unsupervised.utils import preprocess_batch_order
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class UxClassifier(torch.nn.Module):
    """
    An Unsupervised eXtended classifier network, composed of a backbone, an
    unsupervised objective head, and a (supervised) classifier head.
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
        super().__init__()

        assert backbone is not None
        assert classifier_head is not None
        assert unsupervised_head is not None

        self.backbone: Backbone = backbone
        self.classifier_head: ClassifierHead = classifier_head
        self.unsupervised_head: UnsupervisedHead = unsupervised_head

        self.input_format: Optional[str] = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.pixel_mean: torch.Tensor
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.pixel_std: torch.Tensor
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))  # type: ignore[call-arg]

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Dict[str, Any]:
        backbone: Backbone = build_backbone(cfg)
        out_shape: Dict[str, ShapeSpec] = backbone.output_shape()
        classifier_head = build_classifier_head(cfg, out_shape)
        unsupervised_head = build_unsupervised_head(cfg, out_shape)
        return {
            "backbone": backbone,
            "classifier_head": classifier_head,
            "unsupervised_head": unsupervised_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
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
                classes; 2) an "unsupervised" key of unspecified (but likely
                :class:`torch.Tensor`-or-`None`) type.
        """
        if not self.training:
            return self.inference(batched_inputs, normalize)

        images = self.preprocess_image(batched_inputs)
        targets = self.preprocess_target(batched_inputs)
        features: Dict[str, torch.Tensor] = self.backbone(images.tensor)

        losses: Dict[str, torch.Tensor] = {}
        classifier_losses: Dict[str, torch.Tensor]
        _, classifier_losses = self.classifier_head(features, targets=targets)

        unsupervised_losses: Dict[str, torch.Tensor]
        _, unsupervised_losses = self.unsupervised_head(
            features, {"inputs": batched_inputs, "images": images}
        )
        losses.update(classifier_losses)
        losses.update(unsupervised_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, Any]],
        do_unsupervised: bool = True,
        do_postprocess: bool = True,
    ) -> List[Dict[str, Any]]:
        images = self.preprocess_image(batched_inputs)
        features: Dict[str, torch.Tensor] = self.backbone(images.tensor)

        class_scores_results: Optional[List[torch.Tensor]]
        if isinstance(self.classifier_head, torch.nn.Module):
            class_scores_raw, _ = self.classifier_head(features)
            class_scores_results = [scores for scores in class_scores_raw]
        else:
            class_scores_results = None

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
            results = self._postprocess(
                batched_inputs,
                images.image_sizes,
                class_scores_results,
                unsupervised_results,
            )
            return results
        else:
            raise NotImplementedError("not implemeneted; enable post-processing.")

    def _postprocess(
        self,
        batched_inputs: Sequence[Dict[str, Any]],
        images_sizes: Sequence[Tuple[int, int]],
        classifier_results: Optional[Sequence[Optional[torch.Tensor]]],
        unsupervised_results: Optional[Sequence[Optional[torch.Tensor]]],
    ) -> List[Dict[str, Any]]:
        n_inputs = len(batched_inputs)

        if classifier_results is None or unsupervised_results is None:
            nones = (None,) * n_inputs
            if classifier_results is None:
                classifier_results = nones
            if unsupervised_results is None:
                unsupervised_results = nones

        if n_inputs != len(images_sizes):
            raise ValueError(f"length mismatch; {n_inputs=} but {len(images_sizes)=}")
        if n_inputs != len(classifier_results):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(classifier_results)=}"
            )
        if n_inputs != len(unsupervised_results):
            raise ValueError(
                f"length mismatch; {n_inputs=} but {len(unsupervised_results)=}"
            )

        results: List[Dict[str, Any]] = [{} for _ in range(n_inputs)]
        for i in range(len(batched_inputs)):
            image_input = batched_inputs[i]
            image_size = images_sizes[i]
            image_class_scores = classifier_results[i]
            image_unsup_result = unsupervised_results[i]

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

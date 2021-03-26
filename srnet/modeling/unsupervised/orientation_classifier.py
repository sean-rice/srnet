from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
import torch

from srnet.modeling.classifier import ClassifierHead
from srnet.modeling.classifier.classifier_head import CLASSIFIER_HEAD_REGISTRY
from srnet.modeling.common.types import Losses
from srnet.utils._utils import find_cfg_node

from .unsupervised_head import (
    UNSUPERVISED_HEAD_REGISTRY,
    UnsupervisedHead,
    UnsupervisedOutput,
)

__all__ = ["OrientationClassifier"]


@UNSUPERVISED_HEAD_REGISTRY.register()
class OrientationClassifier(UnsupervisedHead):
    ANGLE_TRANSFORMS: Sequence[str] = [
        "index",  # interpret the incoming orientations as already class indices
        "strict",  # convert the orientations to indices from strict class set
        "round",  # convert the orientations to indices, rounding to nearest
    ]

    @configurable
    def __init__(
        self,
        classifier_head: ClassifierHead,
        angle_classes: Sequence[Union[int, float]],
        angle_transform: str,
        output_key: str,
    ):
        super(UnsupervisedHead, self).__init__()

        assert len(angle_classes) == len(set(map(float, angle_classes)))
        assert classifier_head.num_classes == len(angle_classes)
        assert angle_transform in self.ANGLE_TRANSFORMS, f"invalid {angle_transform=}"

        self.classifier: ClassifierHead = classifier_head
        self.angle_classes: Sequence[float] = list(map(float, angle_classes))
        self.angle_tranform: str = angle_transform
        self._output_key: str = output_key

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
        node_path: str = "MODEL.ORIENTATION_CLASSIFIER",
    ) -> Dict[str, Any]:
        node = find_cfg_node(cfg, node_path)
        head_node_path = node.HEAD_NODE
        head_node = find_cfg_node(cfg, head_node_path)

        angle_classes: Sequence[Union[int, float]] = node.ANGLE_CLASSES
        angle_transform: str = node.ANGLE_TRANSFORM

        # allow the head class to build its own args, but overwrite some with
        # our values that we "lift" into the OrientationClassifier CfgNode.
        head_args: Dict[str, Any] = CLASSIFIER_HEAD_REGISTRY.get(
            head_node.NAME
        ).from_config(cfg, input_shape=input_shape, node_path=node.HEAD_NODE)
        head_args["num_classes"] = len(angle_classes)
        head_args["loss_weight"] = node.LOSS_WEIGHT
        head_args["loss_key"] = node.LOSS_KEY
        classifier_head = CLASSIFIER_HEAD_REGISTRY.get(head_node.NAME)(**head_args)
        # classifier_head = build_classifier_head(
        #    cfg, input_shape, node_path=head_node_path
        # )
        assert isinstance(classifier_head, ClassifierHead)
        return {
            "classifier_head": classifier_head,
            "angle_classes": angle_classes,
            "angle_transform": angle_transform,
            "output_key": node.OUTPUT_KEY,
        }

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    @property
    def loss_keys(self) -> Set[str]:
        return self.classifier.loss_keys

    @property
    def output_keys(self) -> Set[str]:
        return {self._output_key}

    def into_per_item_iterable(
        self, network_output: UnsupervisedOutput
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {self._output_key: scores} for scores in network_output[self._output_key]
        ]

    def postprocess(
        self, result: Mapping[str, Optional[torch.Tensor]], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        # no post-processing needed
        return {self._output_key: result.get(self._output_key, None)}

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[UnsupervisedOutput, Losses]:

        orientation_targets: Optional[torch.Tensor] = None
        if self.training:
            assert targets is not None
            orientations: List[float] = [
                d["image_orientation"] for d in targets["inputs"]
            ]
            orientation_targets = self._process_orientations(orientations)
            orientation_targets = orientation_targets.to(
                tuple(features.values())[0].device
            )

        cls_scores, cls_losses = self.classifier(features, targets=orientation_targets)
        results: UnsupervisedOutput = {self._output_key: cls_scores}
        return results, cls_losses

    def _process_orientations(self, orientations: Sequence[float]) -> torch.Tensor:
        if self.angle_tranform == "index":
            assert all(
                map(
                    lambda i: i < self.num_classes and float.is_integer(i), orientations
                )
            )
            return torch.tensor(orientations, dtype=torch.int64)
        elif self.angle_tranform == "strict":
            angleset = torch.tensor(
                self.angle_classes, dtype=torch.float, device="cpu", requires_grad=False
            )
            orientation_tensor = torch.tensor(
                orientations, dtype=torch.float, device="cpu", requires_grad=False
            )
            # matches is an int tensor w/ size (2, N_matches <= len(orientations))
            # matches[0] are the index numbers of the input orientations
            # matches[1] are the indices of the orientation's values in angle_classes
            matches = angleset.eq(orientation_tensor.unsqueeze(dim=1)).nonzero().t()
            # N_matches must be the same size as the number of orientations!
            # otherwise, we got an input angle not in the angle_classes
            if matches.shape[1] != len(orientations):
                failed_indices: Set[int] = set(range(0, len(orientations))) - set(
                    matches[0].tolist()
                )
                bad_values = list((i, orientations[i]) for i in failed_indices)
                raise ValueError(
                    f"got invalid orientations (index, value): {bad_values}"
                )
            return matches[1].detach()
        elif self.angle_tranform == "round":
            raise NotImplementedError()
        raise ValueError(f"invalid {self.angle_transform=}")

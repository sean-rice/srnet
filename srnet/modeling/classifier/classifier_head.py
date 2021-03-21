from abc import ABCMeta, abstractmethod
import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
import torch

from srnet.utils._utils import find_cfg_node

from ..common.types import Losses

__all__ = ["CLASSIFIER_HEAD_REGISTRY", "ClassifierHead", "build_unsupervised_head"]

CLASSIFIER_HEAD_REGISTRY = Registry("CLASSIFIER_HEAD")
CLASSIFIER_HEAD_REGISTRY.__doc__ = """
Registry for classifier heads for models like :class:`Classifier`.
Take feature maps and returns an estimated class/index.
"""


logger = logging.getLogger(__name__)


class ClassifierHead(torch.nn.Module, metaclass=ABCMeta):
    """
    A classifier head that accepts feature maps from the backbone and
    carries out some sort of additional objective.

    NOTE: This abstract class is nice to have for reference, but to achieve an
    adequate amount of flexibility to subclasses, the interface isn't really
    well-defined enough to typecheck. Concrete implementations of functions
    will probably have to disable typechecking for their signatures.
    """

    __call__: Callable[..., Tuple[torch.Tensor, Losses]]

    _loss_weight_default: float = 1.0
    _loss_key_default: str = "loss_classifier"

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: CfgNode, input_shape: Dict[str, ShapeSpec], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def loss_keys(self) -> Set[str]:
        ...

    @abstractmethod
    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Losses]:
        raise NotImplementedError()


def build_classifier_head(
    cfg: CfgNode,
    input_shape: Dict[str, ShapeSpec],
    node_path: str = "MODEL.CLASSIFIER_HEAD",
) -> Optional[ClassifierHead]:
    """
    Create a classifier objective from config.

    Returns:
        ClassifierHead: a :class:`ClassifierHead` instance.
    """
    node = find_cfg_node(cfg, node_path)

    name: Optional[str] = node.NAME
    if name is None:
        return None

    classifier_module = CLASSIFIER_HEAD_REGISTRY.get(name)(
        cfg, input_shape=input_shape, node_path=node_path
    )
    assert isinstance(classifier_module, ClassifierHead)
    return classifier_module

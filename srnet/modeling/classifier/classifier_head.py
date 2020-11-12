from abc import ABCMeta, abstractmethod, abstractproperty
import logging
from typing import Any, Dict, Optional

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
import torch

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

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: CfgNode, input_shape: Optional[ShapeSpec], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def postprocess(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractproperty
    def num_classes(self) -> int:
        raise NotImplementedError()


def build_classifier_head(
    cfg: Any, input_shape: Dict[str, ShapeSpec]
) -> Optional[ClassifierHead]:
    """
    Create a classifier objective from config.

    Returns:
        ClassifierHead: a :class:`ClassifierHead` instance.
    """
    name = cfg.MODEL.CLASSIFIER.NAME
    if name == "None":
        return None

    classifier_module = CLASSIFIER_HEAD_REGISTRY.get(name)(cfg, input_shape=input_shape)
    assert isinstance(classifier_module, ClassifierHead)
    return classifier_module

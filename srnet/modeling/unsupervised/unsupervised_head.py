from abc import abstractmethod
import logging
from typing import Any, Dict, Iterable, Optional

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
import torch

__all__ = ["UNSUPERVISED_HEAD_REGISTRY", "UnsupervisedHead", "build_unsupervised_head"]

UNSUPERVISED_HEAD_REGISTRY = Registry("UNSUPERVISED_HEAD")
UNSUPERVISED_HEAD_REGISTRY.__doc__ = """
Registry for unsupervised heads/objectives for models like :class:`UxRCNN`.
Take feature maps and return some target function.
"""


logger = logging.getLogger(__name__)


class UnsupervisedHead(torch.nn.Module):
    @abstractmethod
    def into_per_item_iterable(
        self, network_output: Dict[str, torch.Tensor], **kwargs: Dict[str, Any]
    ) -> Iterable:
        pass

    @abstractmethod
    @classmethod
    def postprocess(cls, **kwargs: Dict[str, Any]) -> Any:
        pass


def build_unsupervised_head(
    cfg: Any, input_shape: ShapeSpec
) -> Optional[UnsupervisedHead]:
    """
    Create an unsupervised objective from config.

    Returns:
        UnsupervisedHead: a :class:`UnsupervisedHead` instance.
    """
    name = cfg.MODEL.UNSUPERVISED_OBJECTIVE.NAME
    if name == "None":
        return None

    unsup_module = UNSUPERVISED_HEAD_REGISTRY.get(name)(cfg, input_shape)
    assert isinstance(unsup_module, UnsupervisedHead)
    return unsup_module

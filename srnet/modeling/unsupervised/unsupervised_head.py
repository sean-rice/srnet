from abc import abstractmethod
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
import torch

from ..common.types import Losses

__all__ = [
    "UNSUPERVISED_HEAD_REGISTRY",
    "UnsupervisedHead",
    "UnsupervisedOutput",
    "build_unsupervised_head",
]

UNSUPERVISED_HEAD_REGISTRY = Registry("UNSUPERVISED_HEAD")
UNSUPERVISED_HEAD_REGISTRY.__doc__ = """
Registry for unsupervised heads/objectives for models like :class:`UxRCNN`.
Take feature maps and return some target function.
"""

logger = logging.getLogger(__name__)

UnsupervisedOutput = Dict[str, torch.Tensor]


class UnsupervisedHead(torch.nn.Module):
    """
    An unsupervised head that accepts feature maps from the backbone and
    carries out some sort of additional objective.

    NOTE: This abstract class is nice to have for reference, but to achieve an
    adequate amount of flexibility to subclasses, the interface isn't really
    well-defined enough to typecheck. Concrete implementations of functions
    will probably have to disable typechecking for their signatures.
    """

    __call__: Callable[..., Tuple[UnsupervisedOutput, Losses]]

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: CfgNode, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError()

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[UnsupervisedOutput, Losses]:
        raise NotImplementedError()

    @abstractmethod
    def into_per_item_iterable(
        self, network_output: UnsupervisedOutput
    ) -> List[torch.Tensor]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def postprocess(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @property
    @abstractmethod
    def loss_keys(self) -> Set[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_keys(self) -> Set[str]:
        raise NotImplementedError()


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

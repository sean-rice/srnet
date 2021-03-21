from typing import Any, FrozenSet, Optional

from detectron2.data.transforms import AugInput, Augmentation
from fvcore.transforms.transform import Transform
import numpy as np

from .build import AUGMENTATION_REGISTRY

__all__ = ["SrAugInput", "TransformWrapper"]


class SrAugInput(AugInput):
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        super().__init__(image, boxes=boxes, sem_seg=sem_seg)
        self._keyset: FrozenSet[str] = frozenset(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        super().transform(tfm)
        for k in self._keyset:
            v = getattr(self, k)
            apply_name = f"apply_{k}"
            if hasattr(tfm, apply_name):
                setattr(self, k, getattr(tfm, apply_name)(v))


@AUGMENTATION_REGISTRY.register()
class TransformWrapper(Augmentation):
    """
    Trivial augmentation that wraps an already-existing transform.
    """

    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args: Any) -> Transform:
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__

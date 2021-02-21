from typing import Any

from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform

from .build import AUGMENTATION_REGISTRY


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

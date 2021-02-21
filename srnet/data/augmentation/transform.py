from typing import Any, Optional, Sequence, Tuple, Union, cast

from fvcore.transforms.transform import Transform
import numpy as np

from .build import TRANSFORM_REGISTRY

__all__ = ["PadTransform"]


@TRANSFORM_REGISTRY.register()
class PadTransform(Transform):
    """
    Pads a source image with a fixed number of pixels at the front/back of
    the spatial axes (shapes NxWxHxC, HxWxC, or HxW).
    """

    def __init__(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        padding_image: Union[float, int, Sequence[Any]],
        padding_segmentation: int = 0,
        **kwargs: Sequence[Any],
    ):
        super().__init__()
        if not (left >= 0 and right >= 0 and top >= 0 and bottom >= 0):
            raise ValueError(
                f"can't pad with negative values; {left=}, {right=}, {top=}, {bottom=}"
            )

        vars = locals()
        vars.pop("kwargs", None)
        vars.update(kwargs)
        self._set_attributes(vars)
        self.padding_image: Union[float, int, Sequence[Any]]
        self.padding_segmentation: int

    def apply_image(
        self, img: np.ndarray, padding_attr: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """
        if padding_attr is None:
            padding_attr = "padding_image"
        constant_values: Union[int, float, Sequence[Any]] = getattr(self, padding_attr)

        if len(img.shape) == 2:
            # HxW
            assert isinstance(constant_values, (int, float, np.uint8))
            padded = np.pad(
                img,
                ((self.top, self.bottom), (self.left, self.right)),
                mode="constant",
                constant_values=constant_values,
            )
        elif len(img.shape) in {3, 4}:
            # [Nx]HxWxC
            pad_width: Tuple[Tuple[int, int], ...] = (
                (self.top, self.bottom),
                (self.left, self.right),
                (0, 0),
            )
            if len(img.shape) == 4:
                pad_width = ((0, 0),) + pad_width
            padded = np.pad(img, pad_width, mode="constant", constant_values=0)
            if not isinstance(constant_values, Sequence):
                constant_values = (constant_values,) * padded.shape[-1]
                constant_values = cast(Sequence, constant_values)
            for c in range(padded.shape[-1]):
                ####################################################
                # |                                                |#
                # |<----------- [..., :self.top, :, c] ----------->|#
                # |                                                |#
                # ========+---------------------------+============|#
                # |   ^   |                           |     ^      |#
                # | :left |        (original)         |            |#
                # |<     >|         (image)           |<  -right: >|#
                # |   v   |                           |     v      |#
                # ========+---------------------------+============|#
                # |<--------- [..., -self.bottom:, :, c] --------->|#
                # |                                                |#
                ####################################################
                padded[..., : self.top, :, c] = constant_values[c]
                padded[..., -self.bottom :, :, c] = constant_values[c]
                padded[..., self.top : -self.bottom, : self.left, c] = constant_values[
                    c
                ]
                padded[
                    ..., self.top : -self.bottom, -self.right :, c
                ] = constant_values[c]
        else:
            raise ValueError(f"too many dimensions in image. {img.shape=}")
        return padded

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation, padding_attr="padding_segmentation")

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        # TODO: check this! maybe using incorrect (top-left) origin?
        return coords + (self.left, self.top)

    # def inverse(self) -> Transform:
    #    return CropTransform()

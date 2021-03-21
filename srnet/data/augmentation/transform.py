from typing import Any, Optional, Sequence, Tuple, Union

from detectron2.data.transforms import transform
from fvcore.transforms.transform import Transform
import numpy as np

from .build import TRANSFORM_REGISTRY

__all__ = ["PadTransform"]


@TRANSFORM_REGISTRY.register()
class SrPadTransform(Transform):
    """
    Pads a source image with a fixed number of pixels at the front/back of
    the spatial axes (shapes NxWxHxC, HxWxC, or HxW).

    This transform is only needed if padded with a vector value over channels;
    otherwise, prefer the other `fvcore.transforms.transform.PadTransform`.
    """

    def __init__(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        padding_image: Union[float, int, Sequence[Any]],
        padding_segmentation: int = 0,
        **kwargs: Sequence[Any],
    ):
        super().__init__()
        if not (x0 >= 0 and x1 >= 0 and y0 >= 0 and y1 >= 0):
            raise ValueError(
                f"can't pad with negative values; {x0=}, {x1=}, {y0=}, {y1=}"
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
            ndarray: image after applying the transformation.
        """
        if padding_attr is None:
            padding_attr = "padding_image"
        constant_values: Union[int, float, Sequence[Any]] = getattr(self, padding_attr)

        if len(img.shape) == 2:
            # HxW
            assert isinstance(constant_values, (int, float, np.uint8))
            padded = np.pad(
                img,
                ((self.y0, self.y1), (self.x0, self.x1)),
                mode="constant",
                constant_values=constant_values,
            )
        elif len(img.shape) in (3, 4):
            # [Nx]HxWxC
            pad_width: Tuple[Tuple[int, int], ...] = (
                (self.y0, self.y1),
                (self.x0, self.x1),
                (0, 0),
            )
            if len(img.shape) == 4:
                pad_width = ((0, 0),) + pad_width
            if not isinstance(constant_values, Sequence):
                padded = np.pad(
                    img, pad_width, mode="constant", constant_values=constant_values
                )
            else:
                # fill with 0s, then replace with vector values below
                padded = np.pad(img, pad_width, mode="constant", constant_values=0)
                for c in range(padded.shape[-1]):
                    # +------------------------------------------------+
                    # |                       ^                        |
                    # |<----------- [..., :self.y0, :, c] ------------>|
                    # |                       V                        |
                    # ========+---------------------------+============|
                    # |   ^   |                           |     ^      |
                    # |< :x0 >|      (original image)     |<-- -x1: -->|
                    # |   v   |                           |     v      |
                    # ========+---------------------------+============|
                    # |                       ^                        |
                    # |<----------- [..., -self.y1:, :, c] ----------->|
                    # |                       V                        |
                    # +------------------------------------------------+
                    padded[..., : self.y0, :, c] = constant_values[c]
                    padded[..., -self.y1 :, :, c] = constant_values[c]
                    padded[..., self.y0 : -self.y1, : self.x0, c] = constant_values[c]
                    padded[..., self.y0 : -self.y1, -self.x1 :, c] = constant_values[c]
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
        return coords + (self.x0, self.y0)

    # def inverse(self) -> Transform:
    #    return CropTransform()


@transform.RotationTransform.register_type("image_orientation")
def rotate_image_orientation(
    rot_transform: transform.RotationTransform, image_orientation: float
) -> float:
    angle = rot_transform.angle
    assert float.is_integer(angle) and int(angle) in (0, 90, 180, 270, -90, -180, -270)
    return (image_orientation + angle) % 360

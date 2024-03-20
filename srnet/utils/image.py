from typing import Any, Union

from PIL import Image
import numpy as np
import torch


def resize_with_pil(
    example: Union[np.ndarray, torch.Tensor],
    h: int,
    w: int,
    mode: Any = Image.LANCZOS,
) -> np.ndarray:
    """
    Resizes an `ndarray` or `Tensor` image (h, w [,c]) with PIL using the
    specified mode (with the high-quality `Image.LANCZOS` default), and
    returns it as an `ndarray`.
    """
    img: np.ndarray
    if torch.is_tensor(example):
        img = example.to("cpu").numpy()
    else:
        img = example
    assert len(img.shape) in (2, 3)
    img = np.array(Image.fromarray(img).resize((w, h), mode))
    return img

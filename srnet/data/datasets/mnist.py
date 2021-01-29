import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
from torchvision.datasets import MNIST

from srnet.utils._utils import split_list
from srnet.utils.image import resize_with_pil

from ._catalog import SRNET_DATASET_CATALOG, SRNET_METADATA_CATALOG

_MNIST_LABEL_MAP: Dict[str, int] = {label: i for i, label in enumerate(MNIST.classes)}

__all__ = ["register_mnist"]


def _make_mnist_dicts(
    id_base: str,
    dataset_path: pathlib.Path,
    proportion: Optional[float] = None,
    seed: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
    resize_mode: Any = Image.ANTIALIAS,
) -> List[Dict[str, Any]]:

    # loading the data
    data: torch.Tensor
    labels: torch.Tensor
    data, labels = torch.load(str(dataset_path))
    data, labels = data.to("cpu"), labels.to("cpu")
    n_examples, h, w, = data.shape
    assert n_examples == labels.shape[0]

    # select indices
    indices: List[int]
    if proportion is None:
        indices = list(range(n_examples))
    else:
        indices, _ = split_list(list(range(n_examples)), proportion, seed)

    # selecting examples & resizing
    # examples[i]'s index is indices[i]
    examples: List[
        np.ndarray
    ]  # list (n_examples) of ndarray of shape (size_h, size_w, 1)
    if size is not None:
        size_h, size_w = size
        examples = [
            np.expand_dims(
                resize_with_pil(data[i], size_h, size_w, resize_mode), axis=-1
            )
            for i in indices
        ]
    else:
        size_h, size_w = h, w
        examples = [np.expand_dims(data[i].numpy(), axis=-1) for i in indices]
    assert len(examples) == len(indices)

    return [
        {
            "file_name": None,
            "image": examples[i],  # shape/size ([size_h, size_w, 1])
            "image_format": "L",
            "height": examples[i].shape[0],
            "width": examples[i].shape[1],
            "image_id": id_base.format(i=index),
            "class_label": labels[index],
        }
        for i, index in enumerate(indices)
    ]


def register_mnist(
    d2_datasets: Union[str, pathlib.Path],
    proportion: Optional[float] = None,
    seed: Optional[int] = None,
    size: Optional[int] = None,
    size_mode: Optional[str] = None,
) -> None:
    if proportion is not None:
        assert (
            proportion - round(proportion, 3) < 1e-9
        ), f"proportion should only go to 3 decimal digits max; got {proportion=}, try {round(proportion, 3)}"
        proportion = round(proportion, 3)
    else:
        seed = None  # proportion == None means seed is meaningless; remove it

    for subset, file_name in (("train", "training.pt"), ("test", "test.pt")):
        if subset == "test":
            proportion_, seed_ = None, None
        else:
            proportion_, seed_ = proportion, seed

        # building dataset name
        dataset_name: str = f"mnist_{subset}"
        if proportion_ is not None:
            dataset_name += f"_proportion={proportion_}"
            if seed_ is not None:
                dataset_name += f"_seed={seed_}"
        if size is not None:
            dataset_name += f"_size={size}x{size}"

        dataset_path = pathlib.Path(d2_datasets) / "MNIST" / "processed" / file_name

        load_subset = functools.partial(
            _make_mnist_dicts,
            id_base=dataset_name + "_i={i}",
            dataset_path=dataset_path,
            proportion=proportion_,
            seed=seed_,
            size=(size, size) if size is not None else None,
        )

        SRNET_DATASET_CATALOG[dataset_name] = load_subset
        SRNET_METADATA_CATALOG[dataset_name] = {
            "classes": MNIST.classes,
            "evaluator_type": "confusion_matrix",
        }

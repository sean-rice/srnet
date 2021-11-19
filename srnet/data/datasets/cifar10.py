import functools
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from srnet.utils._utils import get_datasets_path, split_list
from srnet.utils.image import resize_with_pil

from ._catalog import SRNET_DATASET_CATALOG, SRNET_METADATA_CATALOG

# pixel mean (r,g,b): [125.307, 122.950, 113.865]
# pixel std  (r,g,b): [ 62.993,  62.089,  66.705]
_cifar10 = CIFAR10(root=get_datasets_path() / "CIFAR10", train=False)
_CIFAR10_LABEL_MAP: Mapping[str, int] = {
    label: i for i, label in enumerate(_cifar10.classes)
}

__all__ = ["register_cifar10"]


def _make_cifar10_dicts(
    id_base: str,
    dataset_path: pathlib.Path,
    train: bool,
    proportion: Optional[float] = None,
    seed: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
    resize_mode: Any = Image.ANTIALIAS,
) -> List[Dict[str, Any]]:

    # loading the data
    dataset: CIFAR10 = CIFAR10(root=dataset_path, train=train)

    data: np.ndarray = dataset.data
    labels: List[int] = dataset.targets
    n_examples, h, w, c = data.shape
    assert n_examples == len(labels)

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
    ]  # list (n_examples) of ndarray of shape (size_h, size_w, 3)
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
        examples = [data[i] for i in indices]
    assert len(examples) == len(indices)

    return [
        {
            "file_name": None,
            "image": examples[i],  # shape/size ([size_h, size_w, 3])
            "image_format": "RGB",
            "height": examples[i].shape[0],
            "width": examples[i].shape[1],
            "image_id": id_base.format(i=index),
            "class_label": labels[index],  # `int`
        }
        for i, index in enumerate(indices)
    ]


def register_cifar10(
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

    for subset, is_train in (("train", True), ("test", False)):
        if subset == "test":
            proportion_, seed_ = None, None
        else:
            proportion_, seed_ = proportion, seed

        # building dataset name
        dataset_name: str = f"cifar10_{subset}"
        if proportion_ is not None:
            dataset_name += f"_proportion={proportion_}"
            if seed_ is not None:
                dataset_name += f"_seed={seed_}"
        if size is not None:
            dataset_name += f"_size={size}x{size}"

        dataset_path = pathlib.Path(d2_datasets) / "CIFAR10"

        load_subset = functools.partial(
            _make_cifar10_dicts,
            id_base=dataset_name + "_i={i}",
            dataset_path=dataset_path,
            train=is_train,
            proportion=proportion_,
            seed=seed_,
            size=(size, size) if size is not None else None,
        )

        SRNET_DATASET_CATALOG[dataset_name] = load_subset
        SRNET_METADATA_CATALOG[dataset_name] = {
            "classes": _cifar10.classes,
            "evaluator_type": "confusion_matrix",
        }

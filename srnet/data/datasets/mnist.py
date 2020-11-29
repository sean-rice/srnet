import functools
import pathlib
from typing import Any, Dict, List, Union

import torch
from torchvision.datasets import mnist

from ._catalog import SRNET_DATASET_CATALOG, SRNET_METADATA_CATALOG

_MNIST_LABEL_MAP: Dict[str, int] = {
    label: i for i, label in enumerate(mnist.MNIST.classes)
}


def _make_mnist_dicts(id_base: str, dataset_path: pathlib.Path) -> List[Dict[str, Any]]:
    data: torch.Tensor
    labels: torch.Tensor
    data, labels = torch.load(str(dataset_path))
    data, labels = data.to("cpu"), labels.to("cpu")
    n_examples, h, w, = data.shape
    assert n_examples == labels.shape[0]
    return [
        {
            "file_name": None,
            "image": example.unsqueeze(dim=-1).numpy(),  # torch.Size([28, 28, 1])
            "image_format": "L",
            "height": h,
            "width": w,
            "image_id": id_base.format(i=i),
            "class_label": label,
        }
        for i, (example, label) in enumerate(zip(data, labels))
    ]


def register_mnist(d2_datasets: Union[str, pathlib.Path], name: str = "mnist") -> None:
    for subset, file_name in (("train", "training.pt"), ("test", "test.pt")):
        dataset_name = f"{name}_{subset}"
        dataset_path = pathlib.Path(d2_datasets) / "MNIST" / "processed" / file_name

        load_subset = functools.partial(
            _make_mnist_dicts, id_base=dataset_name + "_{i}", dataset_path=dataset_path
        )

        SRNET_DATASET_CATALOG[dataset_name] = load_subset
        SRNET_METADATA_CATALOG[dataset_name] = {
            "classes": mnist.MNIST.classes,
            "evaluator_type": "confusion_matrix",
        }

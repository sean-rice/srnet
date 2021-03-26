from collections import OrderedDict
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from detectron2.data import Metadata
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import torch

__all__ = ["ConfusionMatrixDatasetEvaluator"]


class ConfusionMatrixDatasetEvaluator(DatasetEvaluator):
    NAMES: Sequence[str] = ("predicted", "actual")

    def __init__(
        self,
        num_classes: int,
        metadata: Optional[Metadata] = None,
        distributed: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._metadata: Optional[Metadata] = metadata
        self._distributed: bool = distributed
        self._output_dir: Optional[str] = output_dir

        self._cm: torch.Tensor = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long,
            device="cpu",
            requires_grad=False,
        )

    def reset(self) -> None:
        self._cm = torch.zeros(self._cm.size(), out=self._cm)

    def process(
        self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]
    ) -> None:
        """
        Args:
            inputs: the inputs to a :class:`Classifier`-like model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a :class:`Classifier`-like model. It is a
                list of dicts with key "pred_class_scores" that contains a
                `torch.Tensor` of class scores.
        """
        for input, output in zip(inputs, outputs):
            actual_class: int = input["class_label"].item()
            predicted_class: int = int(torch.argmax(output["pred_class_scores"]).item())
            self._cm[predicted_class, actual_class] += 1

    def evaluate(self, max_table_size: int = 25) -> "OrderedDict[str, Dict[str, Any]]":
        # if distributed, gather and sum confusion matrices
        cm: torch.Tensor
        if self._distributed:
            comm.synchronize()
            cms = comm.gather(self._cm, dst=0)
            if not comm.is_main_process():
                return OrderedDict()
            cm = torch.stack(cms, dim=0).sum(dim=0)
        else:
            cm = self._cm

        # saving confusion matrix
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "confusion_matrix.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(cm, f)

        # calculating accuracy
        accuracy = self.accuracy(cm)

        # displaying confusion matrix as table (if it isn't too huge)
        if self.num_classes <= max_table_size:
            headers, showindex = (), "default"
            if self._metadata is not None:
                headers = self._metadata.get("classes", default=headers)
                showindex = self._metadata.get("classes", default=showindex)

            table = tabulate(
                cm,
                headers=headers,
                showindex=showindex,
                tablefmt="pipe",
                floatfmt=".0f",
                numalign="left",
            )
            self._logger.info(table)

        # collect and return results
        results: OrderedDict[str, Dict[str, float]] = OrderedDict(
            [("classification", {"top1": accuracy})]
        )
        self._logger.info(results)
        return results

    @property
    def num_classes(self) -> int:
        return self._cm.shape[0]

    @classmethod
    def accuracy(cls, cm: torch.Tensor) -> float:
        n_correct: int = int(cm.diag().sum().item())
        n_total: int = int(cm.sum().item())
        accuracy = n_correct / n_total
        return accuracy

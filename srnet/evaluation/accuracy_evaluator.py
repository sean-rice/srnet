from collections import OrderedDict
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import torch


class AccuracyDatasetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        top_ks: Sequence[int] = (1,),
        distributed: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._distributed: bool = distributed
        self._output_dir: Optional[str] = output_dir

        self.top_ks: Tuple[int, ...] = tuple(sorted(set(k for k in top_ks)))
        self.total: int = 0
        self.correct_in_top_k: Dict[int, int] = {k: 0 for k in self.top_ks}

    def reset(self) -> None:
        self.total = 0
        self.correct_in_top_k = {k: 0 for k in self.top_ks}

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
            # correct classification: predicted_rank == 1, etc.
            predicted_rank: int = int(
                torch.nonzero(
                    (
                        torch.argsort(output["pred_class_scores"], descending=True)
                        == actual_class
                    ).to(torch.uint8)
                ).item()
            ) + 1
            for k in self.top_ks:
                if predicted_rank <= k:
                    self.correct_in_top_k[k] += 1
            self.total += 1

    def evaluate(
        self, max_table_size: int = 25, output_filename: str = "topk_accuracies.pkl"
    ) -> "OrderedDict[str, Dict[str, Any]]":
        total: int
        correct_in_top_k: Dict[int, int]
        # if distributed, gather and sum correct answers
        if self._distributed:
            comm.synchronize()
            correct_in_top_k = {k: 0 for k in self.top_ks}
            totals: List[int] = comm.gather(self.total, dst=0)
            cms: List[Dict[int, int]] = comm.gather(self.correct_in_top_k, dst=0)
            if not comm.is_main_process():
                return OrderedDict()
            else:
                total = sum(totals)
                # merge count dictionaries
                for d in cms:
                    for k, count in d.items():
                        correct_in_top_k[k] += count
        else:
            total = self.total
            correct_in_top_k = self.correct_in_top_k

        # normalize into accuracies:
        accuracies: Dict[int, float] = {
            k: v / total for k, v in correct_in_top_k.items()
        }
        del correct_in_top_k

        # saving accuracies
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, output_filename)
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(accuracies, f)

        # displaying accuracies by k as table (if it isn't too huge)
        if len(self.top_ks) <= max_table_size:
            table = tabulate(
                {str(k): (v,) for k, v in accuracies.items()},
                headers="keys",
                showindex="default",
                floatfmt=".3f",
                numalign="left",
            )
            self._logger.info(table)

        # collect and return results
        results: OrderedDict[str, Dict[str, float]] = OrderedDict(
            [("classification", {f"top{k}": v for k, v in accuracies.items()})]
        )
        self._logger.info(results)
        return results

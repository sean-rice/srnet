from collections import OrderedDict
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from fvcore.common.file_io import PathManager
import torch
import torch.nn.functional

__all__ = ["MeanSquaredErrorDatasetEvaluator"]


class MeanSquaredErrorDatasetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        task_name: str,
        n_bins: int = 50,
        get_target: Callable[[Dict[str, Any]], torch.Tensor] = lambda inp: inp["image"],
        get_prediction: Callable[[Dict[str, Any]], torch.Tensor] = lambda outp: outp[
            "unsupervised"
        ]["decoded_image"],
        target_processor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        prediction_processor: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        distributed: bool = True,
        output_dir: Optional[str] = None,
    ):
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._distributed: bool = distributed
        self._output_dir = output_dir

        self._task_name: str = task_name
        self._n_bins: int = n_bins
        self._get_target: Callable[[Dict[str, Any]], torch.Tensor] = get_target
        self._get_prediction: Callable[[Dict[str, Any]], torch.Tensor] = get_prediction
        self._target_processor: Callable[
            [torch.Tensor], torch.Tensor
        ] = target_processor
        self._prediction_processor: Callable[
            [torch.Tensor], torch.Tensor
        ] = prediction_processor

        self._mses: List[float] = []

    def reset(self) -> None:
        self._mses = []

    def process(
        self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]
    ) -> None:
        """
        Args:
            inputs: the mapped inputs to a :class:`UxClassifier`-like model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a :class:`UxClassifier`-like model. It is a
                list of dicts with key "pred_class_scores" that contains a
                `torch.Tensor` of class scores.
        """
        for input, output in zip(inputs, outputs):
            target: torch.Tensor = self._get_target(input)  # HWC image
            prediction: torch.Tensor = self._get_prediction(output)  # HWC image

            target = self._target_processor(target)  # (move, etc)
            prediction = self._prediction_processor(prediction)  # (denormalize, etc?)
            mse: float = torch.nn.functional.mse_loss(
                prediction, target, reduction="mean"
            ).item()
            self._mses.append(mse)

    def evaluate(
        self, output_filename: Optional[str] = None
    ) -> "OrderedDict[str, Dict[str, Any]]":
        mses: List[float]
        # if distributed, gather and sum correct answers
        if self._distributed:
            comm.synchronize()
            mses_lists: List[List[float]] = comm.gather(self._mses, dst=0)
            if not comm.is_main_process():
                return OrderedDict()
            mses = sum(mses_lists, [])  # List[List[float]] -> List[float]
        else:
            mses = self._mses

        mse_tensor: torch.Tensor = torch.as_tensor(
            mses, dtype=torch.float, device=torch.device("cpu")
        )
        del mses

        total_mse: float = mse_tensor.sum().item()

        # saving total mse + histogram
        if self._output_dir:
            if output_filename is None:
                output_filename = f"{self._task_name}_mse_evaluation.json"
            json_dict: Dict[str, Any] = {"mse": total_mse}
            if self._n_bins > 1:
                mn: float = mse_tensor.min().item()
                mx: float = mse_tensor.max().item()
                mse_hist: torch.Tensor = torch.histc(
                    mse_tensor, bins=self._n_bins, min=mn, max=mx
                )
                mse_bins: torch.Tensor = torch.linspace(
                    start=mn, end=mx, steps=self._n_bins
                )
                json_dict["hist_counts"] = mse_hist.tolist()
                json_dict["hist_bins"] = mse_bins.tolist()

            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, output_filename)
            with PathManager.open(file_path, "w") as f:
                json.dump(json_dict, f)

        # collect and return results
        results: "OrderedDict[str, Dict[str, float]]" = OrderedDict(
            [(f"mse_{self._task_name}", {f"mse": total_mse})]
        )
        self._logger.info(results)
        return results

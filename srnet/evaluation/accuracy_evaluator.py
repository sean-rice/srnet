from typing import Any, Dict, Iterable, List, Tuple

from detectron2.evaluation import DatasetEvaluator
import torch


class AccuracyDatasetEvaluator(DatasetEvaluator):
    def __init__(self, top_ks: Iterable[int] = (1,)) -> None:
        self.top_ks: Tuple[int, ...] = tuple(sorted(set(k for k in top_ks)))
        self.correct_with_k: Dict[int, int] = {}
        raise NotImplementedError()

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
            class_ranks = torch.argsort(output["pred_class_scores"])
            raise NotImplementedError()
            # self._cm[predicted_class, actual_class] += 1

from collections import OrderedDict
import itertools
import os
import pickle
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from detectron2.utils import comm
import foolbox
from fvcore.common.file_io import PathManager
from tabulate import tabulate
import torch

from srnet.evaluation.accuracy_evaluator import AccuracyDatasetEvaluator
from srnet.foolbox import foolbox_utils
import srnet.foolbox.model
from srnet.foolbox.wrappers.classifier import FoolboxWrappedClassifier

__all__ = ["FoolboxAccuracyDatasetEvaluator"]


class FoolboxAccuracyDatasetEvaluator(AccuracyDatasetEvaluator):
    def __init__(
        self,
        fmodel: srnet.foolbox.model.FlexiblePyTorchModel,
        attack: foolbox.Attack,
        epsilons: Sequence[float],
        batch_size: int = 8,
        distributed: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        assert isinstance(fmodel._pytorch_module, FoolboxWrappedClassifier)
        super().__init__(top_ks=(1,), distributed=distributed, output_dir=output_dir)

        self._fmodel: srnet.foolbox.model.FlexiblePyTorchModel = fmodel
        self._fattack: foolbox.Attack = attack
        self._epsilons: Tuple[float, ...] = tuple(sorted(set(epsilons)))
        self._batch_size: int = batch_size
        self._correct_by_epsilon: Dict[float, int] = {eps: 0 for eps in self._epsilons}

    def reset(self) -> None:
        super().reset()
        self._fmodel.unstore_call_args(True, None)
        self._correct_by_epsilon = {eps: 0 for eps in self._epsilons}

    def process(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            inputs: the mapped inputs to a :class:`Classifier`-like model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a :class:`Classifier`-like model. It is a
                list of dicts with key "pred_class_scores" that contains a
                `torch.Tensor` of class scores.
            batch_size (int, optional): the size of the batches to use when
                calculating accuracies and performing adversarial attacks. if
                `None`, defaults to this instance's value from initialization.
        """
        bsize: int = self._batch_size if batch_size is None else batch_size
        for b_input, b_output in zip(
            self.batch_iter(inputs, bsize), self.batch_iter(outputs, bsize)
        ):
            # process normal, unperturbed accuracy (effectively epsilon = 0)
            super().process(b_input, b_output)

            # adversarial attack
            with torch.set_grad_enabled(True):
                self._fmodel.store_call_args(
                    True, batched_inputs=b_input, normalize=True
                )
                image_list, targets = self._fmodel._pytorch_module.prepare_forward(
                    mapper=lambda x: x,  # inputs already batched (mapped), not raw loaded
                    loaded_inputs=b_input,  # use batched_inputs
                    normalize=False,  # we normalize in forward
                )
                images_tensor: torch.Tensor = image_list.tensor.to(torch.float)
                raw, clipped, is_adv = self._fattack(
                    self._fmodel, images_tensor, targets, epsilons=self._epsilons
                )
                self._fmodel.unstore_call_args(True, None)
            successes = foolbox_utils.get_successes(is_adv, epsilons=self._epsilons)
            for eps, success_set in successes.items():
                # note: in last batch, len(b_input) <= bsize!
                # thus it's important to use len(b_input), not bsize
                self._correct_by_epsilon[eps] += len(b_input) - len(success_set)

    def evaluate(
        self,
        max_table_size: int = 25,
        output_filename: str = "adversarial_accuracies.pkl",
    ) -> "OrderedDict[str, Dict[str, Any]]":
        total: int
        correct_in_top_k: Dict[int, int]
        correct_by_epsilon: Dict[float, int]
        # if distributed, gather and sum correct answers
        if self._distributed:
            comm.synchronize()
            totals: List[int] = comm.gather(self.total, dst=0)
            citks: List[Dict[int, int]] = comm.gather(self.correct_in_top_k, dst=0)
            cbes: List[Dict[float, int]] = comm.gather(self._correct_by_epsilon, dst=0)
            if not comm.is_main_process():
                return OrderedDict()
            else:
                correct_in_top_k = {k: 0 for k in self.top_ks}
                correct_by_epsilon = {eps: 0 for eps in self._epsilons}
                total = sum(totals)
                # merge count dictionaries
                for d_citk in citks:
                    for k, count in d_citk.items():
                        correct_in_top_k[k] += count
                for d_cbe in cbes:
                    for eps, count in d_cbe.items():
                        correct_by_epsilon[eps] += count
        else:
            total = self.total
            correct_in_top_k = self.correct_in_top_k
            correct_by_epsilon = self._correct_by_epsilon

        # normalize into accuracies:
        accuracies_by_epsilon: Dict[float, float] = {}
        accuracies_by_epsilon[0.0] = correct_in_top_k[1] / total
        for eps, n_correct in correct_by_epsilon.items():
            accuracies_by_epsilon[eps] = n_correct / total
        del correct_in_top_k, correct_by_epsilon

        # saving accuracies
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, output_filename)
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(accuracies_by_epsilon, f)

        # displaying accuracies by k as table (if it isn't too huge)
        if len(self.top_ks) <= max_table_size:
            table = tabulate(
                {
                    self._floatstring(eps): (acc,)
                    for eps, acc in accuracies_by_epsilon.items()
                },
                headers="keys",
                showindex="default",
                floatfmt=".3f",
                numalign="left",
            )
            self._logger.info(table)

        # collect and return results
        results: OrderedDict[str, Dict[str, float]] = OrderedDict(
            [
                ("classification", {"top1": accuracies_by_epsilon[0.0]}),
                (
                    "classification_robustness",
                    {
                        self._floatstring(eps): acc
                        for eps, acc in accuracies_by_epsilon.items()
                    },
                ),
            ],
        )
        self._logger.info(results)
        return results

    @staticmethod
    def batch_iter(it: Iterable, batch_size: int) -> Iterator:
        # by "senderle" on stackoverflow (https://stackoverflow.com/a/22045226)
        it = iter(it)
        return iter(lambda: tuple(itertools.islice(it, batch_size)), ())

    @staticmethod
    def _floatstring(f: float, n_digits: int = 4) -> str:
        fmtstr = "{f:." + str(n_digits) + "f}"
        return fmtstr.format(f=f).rstrip("0").rstrip(".")

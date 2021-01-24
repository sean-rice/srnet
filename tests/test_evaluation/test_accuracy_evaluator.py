import unittest

import torch

from srnet.evaluation.accuracy_evaluator import AccuracyDatasetEvaluator

__all__ = ["TestAccuracyDatasetEvaluator"]


class TestAccuracyDatasetEvaluator(unittest.TestCase):

    # TODO: add more tests!

    def test_accuracy_evaluator_basic(self) -> None:
        io_pairs = [
            (0, [1.0, 0.3, 0.5, 0.6]),  # rank = 1
            (0, [0.9, -17.0, 4.0, 0.0]),  # rank = 2
            (1, [0.0, 10.0, -2.0, 0.2]),  # rank = 1
            (1, [13.2, 9.5, 10.3, 5.8]),  # rank = 3
        ]

        fake_inputs = [{"class_label": torch.tensor((c,))} for c, _ in io_pairs]
        fake_outputs = [{"pred_class_scores": torch.tensor(s)} for _, s in io_pairs]

        evaluator12 = AccuracyDatasetEvaluator(
            [1, 2, 3], distributed=False, output_dir=None
        )
        evaluator12.process(fake_inputs, fake_outputs)
        results = evaluator12.evaluate(max_table_size=0)

        for k, correct in zip([1, 2, 3], [2 / 4, 3 / 4, 1.0]):
            k_acc = results["classification"][f"top{k}"]
            self.assertTrue(k_acc == correct, msg=f"{k=}, {k_acc=}, {correct=}")

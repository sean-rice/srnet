from typing import Dict, List, Mapping, Sequence, Set
import unittest

import torch

from srnet.foolbox import foolbox_utils


class TestImageListUtils(unittest.TestCase):
    @staticmethod
    def _build_adv(
        true_successes: Mapping[float, Set[int]],
        epsilons: Sequence[float],
        batch_size: int,
    ) -> torch.Tensor:
        # check args
        for eps, i_successes in true_successes.items():
            for i_success in i_successes:
                if i_success >= batch_size:
                    raise ValueError(f"buggy test: got {i_success=} >= {batch_size=}")
        # build tensor that would produce true_successes
        is_adv: torch.Tensor = torch.full(
            (len(epsilons), batch_size), False, dtype=torch.bool
        )
        for ieps, eps in enumerate(epsilons):
            if eps in true_successes:
                for isuccess in true_successes[eps]:
                    is_adv[ieps, isuccess] = True
        return is_adv

    def test_get_successes_none(self) -> None:
        epsilons: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        true_successes: Dict[float, Set[int]] = {eps: set() for eps in epsilons}
        is_adv: torch.Tensor = TestImageListUtils._build_adv(
            true_successes=true_successes, epsilons=epsilons, batch_size=32
        )
        successes = foolbox_utils.get_successes(is_adv=is_adv, epsilons=epsilons)
        self.assertTrue(true_successes == successes)

    def test_get_successes_none_batch_size_1(self) -> None:
        epsilons: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        true_successes: Dict[float, Set[int]] = {eps: set() for eps in epsilons}
        is_adv: torch.Tensor = TestImageListUtils._build_adv(
            true_successes=true_successes, epsilons=epsilons, batch_size=1
        )
        successes = foolbox_utils.get_successes(is_adv=is_adv, epsilons=epsilons)
        self.assertTrue(true_successes == successes)

    def test_get_successes(self, batch_size: int = 32) -> None:
        epsilons: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5]
        true_successes: Dict[float, Set[int]] = {
            1.0: set(),
            2.0: {0},
            3.0: set(range(0, batch_size)),
            4.0: set(
                x for x in (1, 3, 5, 8, 9, 12, 27, 63, 64, 99, 200) if x < batch_size
            ),
            5.0: set(x for x in range(0, batch_size, 4)),
            6.0: {batch_size - 1},
            7.0: {x for x in range(16, 32) if x < batch_size},
            7.5: {x for x in range(1024, 2048) if x < batch_size},
        }
        is_adv: torch.Tensor = TestImageListUtils._build_adv(
            true_successes=true_successes, epsilons=epsilons, batch_size=batch_size
        )
        successes = foolbox_utils.get_successes(is_adv=is_adv, epsilons=epsilons)
        self.assertTrue(true_successes == successes)

    def test_get_successes_batch_size_1(self) -> None:
        self.test_get_successes(batch_size=1)

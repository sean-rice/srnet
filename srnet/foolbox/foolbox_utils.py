from typing import Dict, Sequence, Set

import torch

__all__ = ["get_successes"]


def get_successes(
    is_adv: torch.Tensor, epsilons: Sequence[float]
) -> Dict[float, Set[int]]:
    """
    Returns the set of indicies in which a Foolbox attack has indicated that it
    has sucessfully found an adversarial attack perturbation.
    """
    if is_adv.shape[0] != len(epsilons):
        raise ValueError(f"mismatch between {is_adv.shape[0]} and {len(epsilons)}")

    is_adv = is_adv.to(device=torch.device("cpu"))  # [Neps, N]

    success_for_epsilon: Dict[float, Set[int]] = {}
    for adv_for_epsilon, epsilon in zip(is_adv, epsilons):
        successes: Set[int] = {
            i for i, result in enumerate(adv_for_epsilon) if result.item() == True
        }
        success_for_epsilon[epsilon] = successes
    return success_for_epsilon

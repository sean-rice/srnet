import copy
import random
from typing import Any, List, Sequence, Tuple

__all__ = ["deepmap", "all_to", "random_split_list"]


def deepmap(inputs: Any, function: Any, qualifier: Any) -> Any:
    """
    Maps a function over an iterable input (deep structure of `Sequence`s,
    `dict`s, or `set`s). When the function application occurs (the "leaves" of
    the traversal) is determined with the qualifier function parameter.
    """
    if qualifier(inputs):
        return function(inputs)

    def recurse(i):
        return deepmap(i, function, qualifier)

    if isinstance(inputs, Sequence):
        return type(inputs)(map(recurse, inputs))  # type: ignore[call-arg]
    elif isinstance(inputs, dict):
        return type(inputs)({k: recurse(v) for k, v in inputs.items()})
    elif isinstance(inputs, set):
        return type(inputs)({recurse(v) for v in inputs})
    raise ValueError(f"unsupported type for inputs: {type(inputs)}")


def all_to(inputs: Any, *args: Any, **kwargs: Any) -> Any:
    r"""
    Calls the `to()` method of all `torch.Tensor`s (or any other objects that
    have a `to()` attribute) in a (potentially deep) input structure with the
    provided arguments.
    """

    def q(i: Any) -> bool:
        return hasattr(i, "to")

    def f(i: Any) -> Any:
        return i.to(*args, **kwargs)

    return deepmap(inputs, f, q)


def random_list_split(
    items: List[Any], seed: int, proportion: float
) -> Tuple[List[Any], List[Any]]:
    """
    Splits a `list` randomly (but deterministically) into two portions.

    Args:
        items (List[Any]): The list of items to split.
        seed (int): The seed used to determine the random split.
        proportion (float): The proportion of items in the "left" split list.
    
    Returns:
        left_list (List[Any]): The "left" split of the list (with `proportion`
            of the original items).
        right_list (List[Any]): The "right" split of the list (with
            `1 - proportion` of the original items).
    """
    items = copy.copy(items)
    random.Random(seed).shuffle(items)
    n_left = int(proportion * len(items))
    left_items = items[:n_left]
    right_items = items[n_left:]
    return left_items, right_items

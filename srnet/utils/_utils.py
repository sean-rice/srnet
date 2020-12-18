import copy
import os
import pathlib
import random
from typing import Any, List, Optional, Sequence, Tuple

from detectron2.config import CfgNode

__all__ = [
    "deepmap",
    "all_to",
    "split_list",
    "find_cfg_node",
    "get_datasets_path",
]


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


def split_list(
    items: List[Any], proportion: float, seed: Optional[int],
) -> Tuple[List[Any], List[Any]]:
    """
    Splits a `list`, optionally randomly (but deterministically), into two
    portions.

    Args:
        items (List[Any]): The list of items to split.
        proportion (float): The proportion of items in the "left" split list.
        seed (int, optional): The seed used to determine the random split. If
            `None`, the split isn't randomized and the split is done with the
            original ordering of the input items.
    
    Returns:
        left_list (List[Any]): The "left" split of the list (with `proportion`
            of the original items).
        right_list (List[Any]): The "right" split of the list (with
            `1 - proportion` of the original items).
    """
    items = copy.copy(items)
    if seed is not None:
        random.Random(seed).shuffle(items)
    n_left = int(proportion * len(items))
    left_items = items[:n_left]
    right_items = items[n_left:]
    return left_items, right_items


def find_cfg_node(cfg: CfgNode, node_path: str) -> CfgNode:
    """
    Finds and returns a target node, given by `node_path`, from a
    :class:`CfgNode` tree.

    Args:
        cfg (CfgNode): The root configuration node to search.
        node_path (str): The name of the desired node to be returned. For
            example, `"MODEL.CLASSIFIER_HEAD"`.
    
    Returns:
        target_node (CfgNode): The `CfgNode` child of the `cfg` corresponding 
            to the provided `node_path`.
    """
    node_names = node_path.split(".")
    target_node = cfg
    for next_node in node_names:
        target_node = getattr(target_node, next_node)
    return target_node


def get_datasets_path(env_var: str = "DETECTRON2_DATASETS") -> pathlib.Path:
    root = os.getenv(env_var)
    if root is None:
        raise EnvironmentError(
            f"environment variable {env_var} is unset; can't determine datasets path"
        )
    root_path = pathlib.Path(root).expanduser()
    return root_path

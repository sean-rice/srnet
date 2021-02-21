from abc import ABCMeta
import inspect
import logging
from typing import Any, Callable, List, Mapping, Optional

from detectron2.config.config import CfgNode
from detectron2.data.transforms import Augmentation, augmentation_impl
from detectron2.data.transforms import transform as d2_transform
from detectron2.utils.registry import Registry
import fvcore.transforms.transform as fvc_transform

from srnet.utils._utils import find_cfg_node

__all__ = [
    "AUGMENTATION_REGISTRY",
    "TRANSFORM_REGISTRY",
    "build_single_augmentation",
    "build_augmentations_from_cfg",
]

AUGMENTATION_REGISTRY = Registry("AUGMENTATION")
AUGMENTATION_REGISTRY.__doc__ = """
Registry for data augmentations. Registered objects should be a subclass of
`detectron2.data.transforms.Augmentation`.
"""
# preload with d2 augmentations
for _, cls in inspect.getmembers(augmentation_impl):
    if type(cls) == type and issubclass(cls, Augmentation):
        AUGMENTATION_REGISTRY.register(cls)

TRANSFORM_REGISTRY = Registry("TRANSFORM")
TRANSFORM_REGISTRY.__doc__ = """
Registry for data transformations. Registered objects should be a subclass of
`fvcore.transforms.transform.Transform`.
"""
# preload with fvcore, d2, srnet transforms
for transform_module in (d2_transform, fvc_transform):
    for _, cls in inspect.getmembers(transform_module):
        if type(cls) in (type, ABCMeta) and issubclass(cls, fvc_transform.Transform):
            try:
                TRANSFORM_REGISTRY.register(cls)
            except:
                pass


def build_single_augmentation(
    name: str, args: Mapping[str, Any], prob: Optional[float] = None
) -> Augmentation:
    if name == "TransformWrapper":
        tfm_args = dict(args)
        tfm_name = tfm_args.pop("_TRANSFORM_NAME")
        tfm_cls = TRANSFORM_REGISTRY.get(tfm_name)
        tfm_instance = tfm_cls(**tfm_args)
        args = {"tfm": tfm_instance}

    aug_cls = AUGMENTATION_REGISTRY.get(name)
    aug = aug_cls(**args)
    if prob is not None:
        aug = augmentation_impl.RandomApply(aug, prob=prob)
    return aug

def _default_pipeline_nodes(cfg: CfgNode, is_train: bool) -> str:
    return "INPUT.AUG.CUSTOM." + ("TRAIN" if is_train else "TEST")

def _build_detectron2_defaults(cfg: CfgNode, is_train: bool) -> List[Augmentation]:
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    d2augs = [augmentation_impl.ResizeShortestEdge(min_size, max_size, sample_style)]
    return d2augs


def build_augmentations_from_cfg(
    cfg: CfgNode,
    is_train: bool,
    node_resolver: Callable[[CfgNode, bool], str] = _default_pipeline_nodes,
    _extras_default: str = "last",
) -> List[Augmentation]:
    """
    Create a list of :class:`Augmentation`s from a config, using the nodes
    determined by a node resolver.

    Returns:
        list[Augmentation]
    """
    _SEQ_NODENAME: str = "SEQUENCE"

    node_path: str = node_resolver(cfg, is_train)
    node: CfgNode = find_cfg_node(cfg, node_path)
    enabled: bool = node.get("ENABLED", False)
    extras: str = node.get("EXTRAS", _extras_default)
    assert extras in ["none", "first", "last"]

    logger = logging.getLogger(__name__)
    augs: List[Augmentation] = []

    # "enabled" determines whether the custom augmentations are built.
    if enabled == True:
        for i, aug_cfg in enumerate(node.get(_SEQ_NODENAME)):
            try:
                aug = build_single_augmentation(
                    name=aug_cfg["NAME"],
                    args=aug_cfg["ARGS"],
                    prob=aug_cfg.get("PROB", None),
                )
            except KeyError:
                logger.error(
                    f"failed to build augmentation: {node_path}.{_SEQ_NODENAME}[{i}]: {aug_cfg}"
                )
                raise
            augs.append(aug)

    # "extras" are the things d2 adds by default.
    # to try to avoid a gotcha for users, "extras" are opt-out with "none"
    if extras != "none":
        extra_augs: List[Augmentation] = _build_detectron2_defaults(cfg, is_train)
        if extras == "first":
            augs = extra_augs + augs
        elif extras == "last":
            augs = augs + extra_augs
        else:
            raise ValueError(f'somehow got invalid value for extras: "{extras}"')
    logger.info(
        f"Augmentations used in {'training' if is_train else 'testing'}: " + str(augs)
    )
    return augs

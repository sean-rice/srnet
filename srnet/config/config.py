from detectron2.config import CfgNode


def add_srnet_config(cfg: "CfgNode") -> "CfgNode":
    r"""
    Adds the configuration options related to srnets to a `CfgNode` (in-place)
    and returns it.

    Args:
        cfg (detectron2.config.CfgNode): The standard detectron2 config.

    Returns:
        cfg (detectron2.config.CfgNode): The modified config ready for `srnet`s.
    """
    raise NotImplementedError()

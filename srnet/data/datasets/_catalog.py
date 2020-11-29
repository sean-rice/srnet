from typing import Any, Callable, Dict

__all__ = ["SRNET_DATASET_CATALOG", "SRNET_METADATA_CATALOG"]

SRNET_DATASET_CATALOG: Dict[str, Callable] = {}
"""
because the detectron2 DatasetCatalog is basically static, the srnet code 
can't instantiate its own (to be kept separate until an explicit merging of
d2 and srnet with `srnet.merge_with_detectron2()`). thus, this module
contains the replacement srnet "global" container to "register" datasets and
zero-argument callables with until an explicit merge is performed.
"""

SRNET_METADATA_CATALOG: Dict[str, Dict[str, Any]] = {}

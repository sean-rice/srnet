from . import config, data, engine, evaluation, foolbox, modeling, tools, utils


def merge_with_detectron2():
    """
    Modifies the mutable parts of detectron2's modules/environment (such as
    `Registry`s, `DatasetCatalog`s, etc.) with the corresponding features from
    `srnet`. This makes them accessible from within `detectron2` code. By
    design, srnet doesn't modify/monkey-patch any part of `detectron2` without
    calling this function.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.modeling.backbone import BACKBONE_REGISTRY as D2_BACKBONE_REGISTRY
    from detectron2.modeling.meta_arch import (
        META_ARCH_REGISTRY as D2_META_ARCH_REGISTRY,
    )
    from detectron2.utils.registry import Registry

    from .data import builtin  # noqa
    from .data.datasets._catalog import SRNET_DATASET_CATALOG, SRNET_METADATA_CATALOG
    from .modeling.backbone.build import BACKBONE_REGISTRY
    from .modeling.meta_arch.build import META_ARCH_REGISTRY

    # registry merge helper function
    def merge_registries(
        primary: Registry, secondary: Registry, overwrite: bool = False
    ) -> Registry:
        primary_map = primary._obj_map
        secondary_map = secondary._obj_map
        for obj_name, obj_ in secondary_map.items():
            if obj_name in primary_map and overwrite == False:
                raise ValueError(
                    f"primary registry {primary._name} already contains {obj_name}, can't add it from registry {secondary._name}!"
                )
            primary_map[obj_name] = obj_
        return primary

    # merging various registries
    merge_registries(D2_META_ARCH_REGISTRY, META_ARCH_REGISTRY)
    merge_registries(D2_BACKBONE_REGISTRY, BACKBONE_REGISTRY)

    # merging dataset catalogs
    for dataset_name, dataset_callable in SRNET_DATASET_CATALOG.items():
        DatasetCatalog.register(dataset_name, dataset_callable)

    # merging metadata catalogs
    for metadata_name, metadata_dict in SRNET_METADATA_CATALOG.items():
        metadata_catalog = MetadataCatalog.get(metadata_name)
        for key, value in metadata_dict.items():
            setattr(metadata_catalog, key, value)

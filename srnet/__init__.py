from detectron2.utils.registry import Registry


def merge_with_detectron2():
    """
    Modifies the mutable parts of detectron2's modules/environment (such as
    `Registry`s, `DatasetCatalog`s, etc.) with the corresponding features from
    `srnet`. This makes them accessible from within `detectron2` code. By
    design, srnet doesn't modify/monkey-patch any part of `detectron2` without
    calling this function.
    """

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

    from detectron2.modeling.meta_arch import (
        META_ARCH_REGISTRY as D2_META_ARCH_REGISTRY,
    )

    from .modeling.meta_arch.build import META_ARCH_REGISTRY

    merge_registries(D2_META_ARCH_REGISTRY, META_ARCH_REGISTRY)

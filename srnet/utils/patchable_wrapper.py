from typing import Any, Set

__all__ = ["PatchableWrapper"]


class PatchableWrapper:
    """
    A base class that acts exactly as if it is actually its wrapped member,
    with the exception of any patched attributes.
    
    The usage of this class is to:
        1) Create a new class `X` that multiple inherits both
        `PatchableWrapper` (as the first superclass) and some other class `T`
        (as the second).

        2) The new `X` class should be instantiated with an object `t: T`, and
        wraps it by calling `super().__init__(t)`, leading to
        `PatchableWrapper.__init__`.
        
        3) Any necessary patches can be made in `X.__init__` after this super
        init call.
        
        4) `X` will behave exactly as if it were actually the object
        `X._wrapped__`, except when getting attributes that have been patched.
    """

    def __init__(self, wrapped):
        self._patched_attrs__: Set[str] = {"_wrapped__"}
        self._wrapped__ = wrapped

    def __getattribute__(self, name: str) -> Any:
        if name in {"_patched_attrs__", "_patch__"}:
            return super().__getattribute__(name)
        elif name in super().__getattribute__("_patched_attrs__"):
            return super().__getattribute__(name)
        return self._wrapped__.__getattribute__(name)

    def _patch__(self, name: str, obj: Any) -> None:
        self._patched_attrs__.add(name)
        setattr(self, name, obj)

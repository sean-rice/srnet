import copy
from typing import Any, Dict, Optional, Union

import foolbox

from srnet.foolbox.model import FlexiblePyTorchModel
from srnet.utils.patchable_wrapper import PatchableWrapper

__all__ = ["FoolboxAttackWrapper"]


class FoolboxAttackWrapper(PatchableWrapper, foolbox.attacks.base.FixedEpsilonAttack):
    """
    Deprecated.
    """

    def __init__(self, attack: foolbox.attacks.base.FixedEpsilonAttack):
        super().__init__(attack)
        self._patch__("run", object.__getattribute__(self, "run"))

    # required to be defined to instantiate FixedEpsilonAttack; not used.
    @property
    def distance(self) -> foolbox.Distance:
        raise NotImplementedError()

    def run(
        self,
        model: foolbox.Model,
        inputs: foolbox.attacks.base.T,
        criterion: Union[foolbox.criteria.Misclassification, foolbox.attacks.base.T],
        *,
        epsilon: float,
        **model_args: Any,  # warning: foolbox will call inference without this as a part of deciding if its attacks were successful
    ) -> foolbox.attacks.base.T:
        old_stored: Optional[Dict[str, Any]] = None
        if (
            isinstance(model, FlexiblePyTorchModel)
            and model_args is not None
            and len(model_args) > 0
        ):
            old_stored = copy.copy(model._stored_call_args)
            model.store_call_args(False, **model_args)
        result = self._wrapped__.run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon
        )
        if old_stored is not None and isinstance(model, FlexiblePyTorchModel):
            model.unstore_call_args(False, None)
            model.store_call_args(True, **old_stored)
        return result

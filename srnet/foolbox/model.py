import copy
from typing import Any, Callable, Dict, Optional, Set, cast

import foolbox
import torch

__all__ = ["FlexiblePyTorchModel"]


class FlexiblePyTorchModel(foolbox.PyTorchModel):
    """
    An extended version of foolbox's `PyTorchModel` that allows for extra
    arguments to be stored and unstored. Useful for integrating with detectron2
    models which may require more inputs than a sole input `torch.Tensor`.
    """

    def __init__(
        self,
        model: Any,
        bounds: foolbox.types.BoundsInput,
        device: Any = None,
        preprocessing: foolbox.types.Preprocessing = None,
        model_callable: Optional[Callable] = None,
    ):
        super().__init__(
            model=model, bounds=bounds, device=device, preprocessing=preprocessing
        )
        if model_callable is not None:
            self._model = model_callable
        self._pytorch_module = model
        self._stored_call_args: Dict[str, Any] = {}

    def store_call_args(self, /, strict: bool, **kwargs: Any) -> None:
        if strict == False:
            self._stored_call_args.update(kwargs)
        else:
            for k, v in kwargs.items():
                if k in self._stored_call_args:
                    raise ValueError(f"can't store call arg {k}; already present!")
                self._stored_call_args[k] = v
        self._update_model_call()

    def unstore_call_args(
        self, strict: bool = False, arguments: Optional[Set[Any]] = None
    ) -> None:
        if arguments is None:
            self._stored_call_args.clear()
        elif strict == False:
            for k in arguments:
                self._stored_call_args.pop(k, None)
        else:
            for k in arguments:
                self._stored_call_args.pop(k)
        self._update_model_call()

    def _update_model_call(self) -> None:
        self._model = self.get_flexible_model_call(
            self._pytorch_module, self._stored_call_args
        )

    @staticmethod
    def get_flexible_model_call(
        model: torch.nn.Module, stored_args: Optional[Dict[str, Any]] = None
    ) -> Callable[..., torch.Tensor]:
        _stored_args = copy.copy(stored_args) if stored_args is not None else {}

        def flexible_model_call(
            input: torch.Tensor, **model_args: Dict[str, Any]
        ) -> torch.Tensor:
            with torch.set_grad_enabled(input.requires_grad):
                d = copy.copy(_stored_args)
                d.update(model_args)
                result = cast(torch.Tensor, model(input, **d))
            return result

        return flexible_model_call

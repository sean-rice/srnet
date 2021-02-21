from typing import Any, Dict, Mapping, Tuple

import torch

__all__ = ["BlockSpecification", "Losses"]

BlockSpecification = Tuple[str, Mapping[str, Any]]  # (class_name, kwargs)
Losses = Dict[str, torch.Tensor]

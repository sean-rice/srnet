from typing import Any, Dict, List, Optional, Tuple

from detectron2.data import DatasetMapper
from detectron2.structures import ImageList
import torch

from srnet.modeling.meta_arch.classifier import Classifier, ClassifierResult
from srnet.utils.image_list import repad_image_list

__all__ = ["FoolboxWrappedClassifier"]


class FoolboxWrappedClassifier(torch.nn.Module):
    """
    A module that wraps a `Classifier` and exposes an interface for use with
    foolbox. Most importantly, the `forward` method is changed to ony have one
    required input-- a raw NCHW tensor-- rather than batched_inputs like a
    more standard detectron2 model.
    """

    def __init__(
        self,
        model: Classifier,
        normalize_in_forward: bool = True,
        repad_value: float = 0.0,
    ):
        super().__init__()
        self.model: Classifier = model
        self.normalize_in_forward: bool = normalize_in_forward
        self.repad_value = repad_value
        self.eval()

    def forward(
        self,
        input_tensor: torch.Tensor,
        batched_inputs: Optional[List[Dict[str, Any]]] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        normalize = self.normalize_in_forward if normalize is None else normalize

        if normalize == True:
            assert batched_inputs is not None
            input_tensor = self.model._normalize(input_tensor)
            il = ImageList(
                input_tensor, [(i["height"], i["width"]) for i in batched_inputs]
            )
            il = repad_image_list(il, pad_value=0.0)
            input_tensor = il.tensor

        self.model._patch_forward(self.model.layers)
        result: ClassifierResult = self.model(input_tensor)
        self.model._patch_forward(None)
        return result.class_scores

    def prepare_forward(
        self,
        mapper: DatasetMapper,
        loaded_inputs: List[Dict[str, Any]],
        normalize: Optional[bool] = None,
    ) -> Tuple[ImageList, torch.Tensor]:
        """
        Applies a DatasetMapper to the dataloader-provided inputs, then
        performs preprocessing on the images and targets.
        """
        normalize = not self.normalize_in_forward if normalize is None else normalize

        batched_inputs = [mapper(input) for input in loaded_inputs]
        image_list: ImageList = self.model.preprocess_image(
            batched_inputs, normalize=normalize
        )
        targets: torch.Tensor = self.model.preprocess_target(batched_inputs)
        return image_list, targets

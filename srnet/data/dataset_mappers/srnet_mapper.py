import copy
from typing import Any, Callable, Dict, Optional

from detectron2.config import CfgNode
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
import numpy as np
import torch

from srnet.data.augmentation.augmentation import SrAugInput

__all__ = ["SrnetDatasetMapper"]


class SrnetDatasetMapper(DatasetMapper):
    """
    An enhanced dataset mapper (callable which takes a dataset dict in
    Detectron2 format and maps it into a format used by the model) that adds
    the following functionality:

        (1) Allows for "pre-loaded" dataset dicts, which are dataset dicts that
            already contain `"image"` and `"image_format"` keys.
            The value for `"image"` should be of type `np.ndarray`, in HWC
            layout. The `"image_format"` should be a string corresponding to
            a PIL `Image` mode.
    
        (2) A configurable function parameter used to build the image
            augmentation pipeline. In `from_config`, the `augmentation_builder`
            parameter is an optional customizable callable that replaces the
            default `DatasetMapper` image augmentations with the returned
            results of the provided callable.
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if "image" in dataset_dict:
            if not isinstance(dataset_dict["image"], np.ndarray):
                raise ValueError(
                    f'in {dataset_dict["image_id"]=}, pre-loaded "image" should be ndarray (HWC), got {type(dataset_dict["image"])}.'
                )
            if len(dataset_dict["image"].shape) != 3:
                raise ValueError(
                    f'in {dataset_dict["image_id"]=}, pre-loaded "image" should be ndarray in HWC format, got {len(dataset_dict["image"].shape)=}.'
                )
            if "image_format" not in dataset_dict:
                raise ValueError(
                    f"in {dataset_dict['image_id']=}, found pre-loaded \"image\" key but no \"image_format\" key."
                )
            if dataset_dict["image_format"] != self.image_format:
                raise ValueError(
                    f"in {dataset_dict['image_id']=}, provided {dataset_dict['image_format']=} doesn't match DatasetMapper {self.image_format=}."
                )
            image = dataset_dict["image"]
        else:
            # USER: Write your own image loading if it's not from a file
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = SrAugInput(image, sem_seg=sem_seg_gt, image_orientation=0.0)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt, image_orientation = (
            aug_input.image,
            aug_input.sem_seg,
            aug_input.image_orientation,
        )
        dataset_dict["image_orientation"] = image_orientation  # _type: float

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        is_train: bool = True,
        augmentation_builder: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        # initialize as before
        args = super().from_config(cfg, is_train=is_train)
        # but replace the augmentations with our own
        if augmentation_builder is not None:
            args["augmentations"] = augmentation_builder(cfg, is_train)
        return args

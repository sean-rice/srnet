import argparse
from collections import OrderedDict
import copy
import logging
import os

from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import default_setup
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils import comm
import torch

import srnet
from srnet.config import add_srnet_config
from srnet.engine.defaults import DefaultTrainer as SrDefaultTrainer
from srnet.evaluation import AccuracyDatasetEvaluator, ConfusionMatrixDatasetEvaluator


class Trainer(SrDefaultTrainer):
    """
    We use the "SrDefaultTrainer" which contains pre-defined default logic for
    standard training workflow.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "classifier_accuracy":
            return AccuracyDatasetEvaluator(
                top_ks=getattr(
                    MetadataCatalog.get(dataset_name), "accuracy_top_ks", (1,)
                ),
                distributed=True,
                output_dir=output_folder,
            )
        elif evaluator_type == "confusion_matrix":
            return ConfusionMatrixDatasetEvaluator(
                num_classes=cfg.MODEL.CLASSIFIER_HEAD.NUM_CLASSES,
                metadata=MetadataCatalog.get(dataset_name),
                distributed=True,
                output_dir=output_folder,
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args: argparse.Namespace) -> CfgNode:
    """
    Create configs and perform basic setups.
    """
    srnet.merge_with_detectron2()
    cfg = get_cfg()
    cfg = add_srnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def remove_arg(ap: argparse.ArgumentParser, option: str) -> argparse.ArgumentParser:
    """
    Removes an argument from an `ArgumentParser`.

    Note: uses a non-public behavior of the `argparse` API.
    """
    ap = copy.deepcopy(ap)
    action = list(filter(lambda a: option in a.option_strings, ap._actions))[0]
    ap._handle_conflict_resolve(None, [(option, action)])  # type: ignore[arg-type]
    return ap

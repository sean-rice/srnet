import argparse
from ast import literal_eval
import copy
import os
from typing import Any, Callable, Iterable, Optional, Tuple

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MapDataset, get_detection_dataset_dicts
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import inference_on_dataset, verify_results
import detectron2.utils.comm as comm
import foolbox
import foolbox.attacks

import srnet
from srnet.config import add_srnet_config
from srnet.data.build import build_batch_data_loader
from srnet.data.dataset_mappers import SrnetDatasetMapper
from srnet.evaluation.foolbox_evaluator import FoolboxAccuracyDatasetEvaluator
from srnet.foolbox.model import FlexiblePyTorchModel
from srnet.foolbox.wrappers.classifier import FoolboxWrappedClassifier

from ._common import Trainer, setup, remove_arg

__all__ = ["main", "get_aversarial_argument_parser", "run"]

def main(args):
    cfg = setup(args)

    if args.adversarial_output_dir is None:
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    else:
        output_dir = args.adversarial_output_dir

    if args.adversarial_batch_size > 0:
        batch_size = args.adversarial_batch_size
    else:
        batch_size = cfg.SOLVER.IMS_PER_BATCH

    # load a model
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    wrapped_model = FoolboxWrappedClassifier(model, normalize_in_forward=True)
    wrapped_model = wrapped_model.to(wrapped_model.model.device)
    fmodel = FlexiblePyTorchModel(
        wrapped_model,
        bounds=(0, 255),
        model_callable=FlexiblePyTorchModel.get_flexible_model_call(wrapped_model),
    )

    # load dataset
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TEST,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=0,
        proposal_files=None,
    )
    dataset = MapDataset(dataset, SrnetDatasetMapper(cfg, is_train=True))
    sampler = InferenceSampler(len(dataset))
    loader = build_batch_data_loader(
        dataset=dataset,
        sampler=sampler,
        total_batch_size=batch_size,
        aspect_ratio_grouping=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,  # always test entire dataset, don't drop any examples
    )
    attack = foolbox.attacks.LinfPGD(steps=args.adversarial_steps)
    res = inference_on_dataset(
        model,
        loader,
        FoolboxAccuracyDatasetEvaluator(
            fmodel,
            attack,
            epsilons=args.adversarial_epsilons,
            batch_size=batch_size,
            output_dir=output_dir,
        ),
    )
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


def _parse_tuple(tuple_str: str, cast: Optional[Callable] = None) -> Tuple[Any, ...]:
    t = literal_eval(tuple_str)
    if not isinstance(t, Iterable):
        t = (t,)
    if cast is not None:
        t = tuple(cast(x) for x in t)
    return t


def add_adversarial_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument(
        "--adversarial-output-dir",
        default=None,
        help="where to output the adversarial evaluation. if None, uses OUTPUTDIR/inference. default: %(default)s",
    )
    ap.add_argument(
        "--adversarial-epsilons",
        type=lambda s: _parse_tuple(s, cast=float),
        default=(2.0, 8.0, 16.0),
        help="the epsilons (perturbation budgets) to attack with.",
    )
    ap.add_argument(
        "--adversarial-steps",
        type=int,
        default=40,
        help="the number of steps the adversary can take. default: %(default)s",
    )
    ap.add_argument(
        "--adversarial-batch-size",
        type=int,
        default=0,
        help="the batch size to use. if nonpositive, uses config. default: %(default)s",
    )
    return ap

def _get_args() -> argparse.Namespace:
    ap: argparse.ArgumentParser = default_argument_parser()
    ap = remove_arg(ap, "--eval-only")
    ap = add_adversarial_arguments(ap)
    args: argparse.Namespace = ap.parse_args()
    args.eval_only = True
    return args

def run(args: Optional[argparse.Namespace]=None) -> None:
    if args is None:
        args = _get_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    run()

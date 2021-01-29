import argparse
from ast import literal_eval
import copy
import os
from typing import Any, Callable, Iterable, Optional, Tuple

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MapDataset, get_detection_dataset_dicts
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import default_argument_parser, default_setup, launch
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

from .train import Trainer


def setup(args):
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


def _remove_arg(args: argparse.ArgumentParser, option: str) -> argparse.ArgumentParser:
    """
    Removes an argument from an `ArgumentParser`.

    Note: uses a non-public behavior of the `argparse` API.
    """
    args = copy.deepcopy(args)
    action = list(filter(lambda a: option in a.option_strings, args._actions))[0]
    args._handle_conflict_resolve(None, [(option, action)])  # type: ignore[arg-type]
    return args


def _get_args() -> argparse.Namespace:
    ap: argparse.ArgumentParser = default_argument_parser()
    ap = _remove_arg(ap, "--eval-only")
    ap.add_argument(
        "--adversarial-output-dir",
        default=None,
        help="where to output the adversarial evaluation.",
    )
    ap.add_argument(
        "--adversarial-epsilons",
        type=lambda s: _parse_tuple(s, cast=float),
        default=(2.0, 8.0, 16.0),
        help="the epsilons (perturbation budget) to attack with.",
    )
    ap.add_argument(
        "--adversarial-steps",
        type=int,
        default=40,
        help="the number of steps the adversary can take.",
    )
    ap.add_argument(
        "--adversarial-batch-size",
        type=int,
        default=0,
        help="the batch size to use. if nonpositive, uses config. default: %(default)s",
    )
    args: argparse.Namespace = ap.parse_args()
    args.eval_only = True
    return args


if __name__ == "__main__":
    args: argparse.Namespace = _get_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

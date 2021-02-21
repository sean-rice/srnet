#!/usr/bin/env python
# Based on detectron2/tools/train_net.py (commit 60d7a1f)
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train models in detectron2 + srnet.

"""

import argparse
from typing import Any, Dict, Optional, Type

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import TrainerBase, default_argument_parser, hooks, launch
from detectron2.evaluation import verify_results
import detectron2.utils.comm as comm

from ._common import Trainer, setup

__all__ = ["main", "run"]


def main(args: Any, trainer_class: Optional[Type] = Trainer) -> Optional[Dict]:
    cfg = setup(args)
    trainer_class = Trainer if trainer_class is None else trainer_class
    assert issubclass(trainer_class, TrainerBase)

    if args.eval_only:
        model = trainer_class.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer_class.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(trainer_class.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = trainer_class(cfg, find_unused_parameters=True)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def run(
    args: Optional[argparse.Namespace] = None, trainer_class: Optional[Type] = Trainer
) -> None:
    if args is None:
        args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, trainer_class),
    )


if __name__ == "__main__":
    run()

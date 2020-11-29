import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine.defaults import DefaultTrainer as _D2DefaultTrainer
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from torch.nn.parallel import DistributedDataParallel

from ..data.dataset_mappers import SrnetDatasetMapper

__all__ = ["DefaultTrainer"]


class DefaultTrainer(_D2DefaultTrainer):
    # Based on detectron2.engine.defaults.DefaultTrainer.__init__ (commit 60d7a1f)
    def __init__(self, cfg, find_unused_parameters=False):
        """
        Args:
            cfg (CfgNode): The training configuration.
            find_unused_parameters (bool): whether or not to enable the
                `find_unused_parameters` option in DDP. Required for certain
                kinds of models.
        """
        super(_D2DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        Calls :func:`detectron2.data.build_detection_train_loader`, but with 
        the custom `srnet` datset mapper `SrnetDatasetMapper` substituted.
        """
        if True:
            mapper = SrnetDatasetMapper(cfg, True, augmentation_builder=None)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            return super().build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        Calls :func:`detectron2.data.build_detection_test_loader`, but with 
        the custom `srnet` datset mapper `SrnetDatasetMapper` substituted.
        """
        if True:
            mapper = SrnetDatasetMapper(cfg, False, augmentation_builder=None)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return super().build_test_loader(cfg, dataset_name)

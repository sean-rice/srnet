# %%
_DEBUG: bool = False
if _DEBUG:
    import debugpy  # noqa

try:
    if not callable(display):  # type: ignore[has-type]
        display = print
except NameError:
    display = print

# %%
import logging
import pathlib
from typing import Any, Dict

from detectron2.config import get_cfg, CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
import ipywidgets
from PIL import Image
import torch

import srnet
from srnet.config import add_srnet_config
from srnet.data.dataset_mappers import SrnetDatasetMapper
from srnet.tools.train import Trainer
from srnet.evaluation import ConfusionMatrixDatasetEvaluator

#%%
srnet.merge_with_detectron2()
logger = logging.getLogger()
logger.setLevel(logging.INFO if not _DEBUG else logging.DEBUG)
logger.addHandler(logging.StreamHandler())
setup_logger()

# %%
def make_config(config_path, config_list=None):
    cfg = add_srnet_config(get_cfg())
    if isinstance(config_path, (str, pathlib.Path)):
        cfg.merge_from_file(config_path)
    config_list = [] if config_list is None else config_list
    cfg.merge_from_list(config_list)
    return cfg


def load_model_for_inference(
    cfg: CfgNode, _load_strict: bool = True
) -> torch.nn.Module:
    """
    Loads a model and prepares it for inference by building it based on `cfg`,
    reading and loading the weights from `cfg.MODEL.WEIGHTS`, and setting the
    model to evaluation mode with `model.eval()` before returning it.
    """
    _model = build_model(cfg)
    with open(cfg.MODEL.WEIGHTS, "rb") as _model_f:
        _loaded_model_dict: Dict[str, Any] = torch.load(_model_f)
        _failed_loads = _model.load_state_dict(
            _loaded_model_dict["model"], strict=_load_strict
        )
    _model.eval()
    return _model


# %%
mnist_test = DatasetCatalog.get("mnist_test")
mnist_train = DatasetCatalog.get("mnist_train")
classifier_cfg = make_config(
    "/scratch/rices6/jobs/20201126_mnist_classifier_debug/config.yaml",
    config_list=[
        "MODEL.WEIGHTS",
        "/scratch/rices6/jobs/20201126_mnist_classifier_debug/model_0000099.pth",
    ]
    + (["MODEL.DEVICE", "cpu"] if not torch.cuda.is_available() else [])
    + (["DATALOADER.NUM_WORKERS", "0"] if _DEBUG else []),
)
classifier = load_model_for_inference(classifier_cfg)
mapper = SrnetDatasetMapper(classifier_cfg, is_train=False)

#%%
@ipywidgets.interact(i_example=ipywidgets.IntSlider(min=0, max=20))
def inference_on_test(i_example=0):
    with torch.no_grad():
        _example = mnist_test[i_example]
        _results = classifier([mapper(_example)])
        _score_ranks = torch.argsort(_results[0]["pred_class_scores"], descending=True)
        _predicted_class = _score_ranks[0].item()

        display(Image.fromarray(_example["image"].squeeze(), mode="L"))
        display(f"actual:    {_example['class_label'].item()}")
        display(f"predicted: {_predicted_class}")
        display("scores: ", tuple(zip(range(0, 10), _results[0]["pred_class_scores"])))


# %%
loader_test = Trainer.build_test_loader(classifier_cfg, "mnist_test")
evaluator_test = ConfusionMatrixDatasetEvaluator(
    classifier.classifier_head.num_classes,
    metadata=MetadataCatalog.get("mnist_test"),
    distributed=False,
    output_dir="/home/rices6/misc/evaluation/test",
)
inference_test = inference_on_dataset(classifier, loader_test, evaluator_test)

# %%

# %%
loader_traintest = Trainer.build_test_loader(classifier_cfg, "mnist_train")
evaluator_traintest = ConfusionMatrixDatasetEvaluator(
    classifier.classifier_head.num_classes,
    metadata=MetadataCatalog.get("mnist_train"),
    distributed=False,
    output_dir="/home/rices6/misc/evaluation/train",
)
inference_traintest = inference_on_dataset(
    classifier, loader_traintest, evaluator_traintest
)

# %%

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
from typing import Optional, Any, Dict

import ipywidgets
from detectron2.config import get_cfg, CfgNode
from detectron2.data import DatasetCatalog, Metadata
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import build_model
from detectron2.utils import visualizer
from detectron2.utils.logger import setup_logger
import numpy as np
from PIL import Image
import torch

import srnet
from srnet.config import add_srnet_config
from srnet.data.dataset_mappers import SrnetDatasetMapper

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


def postprocess_to_image(
    image: torch.Tensor,
    model: Optional[torch.nn.Module] = None,
    denormalize: bool = True,
    to_rgb: bool = True,
    input_format: Optional[str] = None,
) -> np.ndarray:
    """
    Takes the output from UxClassifier and converts it to a displayable image.
    """
    if denormalize == True and model is not None and hasattr(model, "_denormalize"):
        output_image = model._denormalize(
            image,
            mean=model.pixel_mean.clone().to(image.device),  # type: ignore[operator]
            std=model.pixel_std.clone().to(image.device),  # type: ignore[operator]
        )
    else:
        output_image = image
    output_image = (
        output_image.detach()
        .to("cpu")
        .permute(1, 2, 0)
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    if to_rgb is True:
        fmt = input_format if input_format else getattr(model, "input_format", None)
        if fmt is None:
            raise ValueError(
                "can't convert to rgb (input_format not provided as parameter, nor in model)"
            )
        output_image = convert_image_to_rgb(output_image, fmt)
    return output_image


def draw_vis(
    input: Dict[str, Any],
    metadata: Optional[Metadata],
    scale: float = 1.0,
    draw_instances: bool = True,
) -> np.ndarray:
    """
        input (Dict[str, Any]): a dict containing an "image" key (format
            provided by "image_format" key if not RGB) and an optional
            "instances" key to draw instances.
            note: the "instances" should be on cpu.
    """
    img = input["image"]
    if input.get("image_format", "RGB") != "RGB":
        img = convert_image_to_rgb(img, input["image_format"])
    if torch.is_tensor(img):
        img = img.numpy().astype("uint8")

    if draw_instances and "instances" in input:
        vis = visualizer.Visualizer(img, metadata, scale=scale)
        visimg = vis.draw_instance_predictions(input["instances"])
        img = visimg.get_image()
    return img


def load_model_for_inference(
    cfg: CfgNode, _load_strict: bool = True, _map_location=None
) -> torch.nn.Module:
    """
    Loads a model and prepares it for inference by building it based on `cfg`,
    reading and loading the weights from `cfg.MODEL.WEIGHTS`, and setting the
    model to evaluation mode with `model.eval()` before returning it.
    """
    _model = build_model(cfg)
    with open(cfg.MODEL.WEIGHTS, "rb") as _model_f:
        _loaded_model_dict: Dict[str, Any] = torch.load(
            _model_f, map_location=_map_location
        )
        _failed_loads = _model.load_state_dict(
            _loaded_model_dict["model"], strict=_load_strict
        )
    _model.eval()
    return _model


# %%
mnist_test = DatasetCatalog.get("mnist_test")

uxclassifier_cfg = make_config(
    "/scratch/rices6/jobs/20210111_mnist-uxclassification_scratch_fc-100-50_100-784_debugging/config.yaml",
    config_list=[
        "MODEL.WEIGHTS",
        "/scratch/rices6/jobs/20210111_mnist-uxclassification_scratch_fc-100-50_100-784_debugging/model_final.pth",
    ]
    + (["MODEL.DEVICE", "cpu"] if not torch.cuda.is_available() else [])
    + (["DATALOADER.NUM_WORKERS", "0", "MODEL.DEVICE", "cpu"] if _DEBUG else []),
)
uxclassifier_model = load_model_for_inference(
    uxclassifier_cfg, _map_location=torch.device(uxclassifier_cfg.MODEL.DEVICE)
)
mapper = SrnetDatasetMapper(uxclassifier_cfg, is_train=False)

# %%
i_example = 0
example = mnist_test[i_example]
example_image = example["image"]
display(Image.fromarray(example_image.squeeze()))
display(example_image.shape)

#%%
@ipywidgets.interact(i_example=ipywidgets.IntSlider(min=0, max=20))
def inference_on_test(i_example=0):
    with torch.no_grad():
        _example = mnist_test[i_example]
        _results = uxclassifier_model([mapper(_example)])
        _score_ranks = torch.argsort(_results[0]["pred_class_scores"], descending=True)
        _predicted_class = _score_ranks[0].item()
        _recon = postprocess_to_image(
            image=_results[0]["unsupervised"],
            model=uxclassifier_model,
            to_rgb=False,
            input_format="L",
        )

        display(Image.fromarray(_example["image"].squeeze(), mode="L"))
        display(Image.fromarray(_recon.squeeze(), mode="L"))
        display(f"actual:    {_example['class_label'].item()}")
        display(f"predicted: {_predicted_class}")
        display("scores: ", tuple(zip(range(0, 10), _results[0]["pred_class_scores"])))


# %%

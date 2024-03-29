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

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, Metadata
from detectron2.data.detection_utils import convert_image_to_rgb, read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils import visualizer
from detectron2.utils.logger import setup_logger
import numpy as np
from PIL import Image
import torch

import srnet
from srnet.config import add_srnet_config

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
    Takes the output from Ux-R-CNN and converts it to a displayable RGB image.
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


# %%
coco_val = DatasetCatalog.get("coco_2017_val")

uxrcnn_cfg = make_config(
    "/scratch/rices6/jobs/20201112_coco-detection_scratch_uxrcnn_R50FPN_ae-p5_6x/config.yaml",
    config_list=[
        "MODEL.WEIGHTS",
        "/scratch/rices6/jobs/20201112_coco-detection_scratch_uxrcnn_R50FPN_ae-p5_6x/model_0099999.pth",
        "MODEL.DEVICE",
        "cpu",
    ]
    + (["DATALOADER.NUM_WORKERS", "0", "MODEL.DEVICE", "cpu"] if _DEBUG else []),
)
uxrcnn_predictor = DefaultPredictor(uxrcnn_cfg)

# %%
i_example = 0
example = coco_val[i_example]
example_image = read_image(example["file_name"], uxrcnn_predictor.input_format)
display(Image.fromarray(example_image))
display(example_image.shape)

# %%
results = uxrcnn_predictor(example_image)

# %%
results_scale = 1.0

# original image with detection boxes
detection_result = draw_vis(
    {
        "image": example_image,
        "image_format": uxrcnn_predictor.input_format,
        "instances": results["instances"],
    },
    metadata=uxrcnn_predictor.metadata,
    scale=results_scale,
    draw_instances=True,
)
# autoencoder reconstruction image
reconstruction_result = postprocess_to_image(
    results["unsupervised"], model=uxrcnn_predictor.model, to_rgb=True
)

# %%
display(Image.fromarray(detection_result))
display(Image.fromarray(reconstruction_result))

# %%
# iterate
import tqdm

num_iterations = 1024
iterations_output_path = (
    pathlib.Path(
        "/scratch/rices6/misc/iterations/20201112_coco-detection_scratch_uxrcnn_R50FPN_ae-p5_6x/model-0099999/"
    )
    / f"coco-val-{i_example}"
)
iterations_output_path.mkdir(parents=True, exist_ok=True)
iteration_results = [None] * (num_iterations + 1)
for i in tqdm.tqdm(range(0, len(iteration_results)), ascii=True, ncols=60):
    if i == 0:
        iteration_results[i] = example_image
    else:
        _result = uxrcnn_predictor(iteration_results[i - 1])
        _reconstruction = postprocess_to_image(
            _result["unsupervised"], model=uxrcnn_predictor.model, to_rgb=False
        )
        iteration_results[i] = _reconstruction

    _image_rgb = convert_image_to_rgb(
        iteration_results[i], uxrcnn_predictor.input_format
    )
    _pil_image = Image.fromarray(_image_rgb)
    frame_path = iterations_output_path / f"frame-{i:07}.png"
    _pil_image.save(frame_path)

# %%
import ipywidgets


@ipywidgets.interact(i=ipywidgets.IntSlider(min=0, max=len(iteration_results) - 1))
def show_iteration(i: int) -> None:
    image_from_model = iteration_results[i]
    image_rgb = convert_image_to_rgb(image_from_model, uxrcnn_predictor.input_format)
    pil_image = Image.fromarray(image_rgb)
    display(pil_image)


# %%
for iframe in range(0, len(iteration_results)):
    image_from_model = iteration_results[iframe]
    image_rgb = convert_image_to_rgb(image_from_model, uxrcnn_predictor.input_format)
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(
        f"/scratch/rices6/misc/iterations/20201112_coco-detection_scratch_uxrcnn_R50FPN_ae-p5_6x/model-0099999/coco-val-{i_example}/frame-{iframe}.png"
    )

# %%

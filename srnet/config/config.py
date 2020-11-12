from detectron2.config import CfgNode


def add_srnet_config(cfg: "CfgNode") -> "CfgNode":
    r"""
    Adds the configuration options related to srnets to a `CfgNode` (in-place)
    and returns it.

    Args:
        cfg (detectron2.config.CfgNode): The standard detectron2 config.

    Returns:
        cfg (detectron2.config.CfgNode): The modified config ready for `srnet`s.
    """

    # image classifier configuration

    cfg.MODEL.CLASSIFIER = CfgNode()
    cfg.MODEL.CLASSIFIER.NUM_CLASSES = -1
    cfg.MODEL.CLASSIFIER.IN_FEATURES = []  # _type: Union[List[str], str]

    cfg.MODEL.POOLING_CLASSIFIER_HEAD = CfgNode()
    # The type of global pooling to use.
    # potential value(s): "avg" (global avg pooling); "max" (global max pooling)
    cfg.MODEL.POOLING_CLASSIFIER_HEAD.POOL_METHOD = "avg"

    # unsupervised learning configuration

    cfg.MODEL.UNSUPERVISED_OBJECTIVE = CfgNode()
    cfg.MODEL.UNSUPERVISED_OBJECTIVE.NAME = ""

    cfg.MODEL.IMAGE_DECODER = CfgNode()
    cfg.MODEL.IMAGE_DECODER.LOSS_WEIGHT = 1.0
    cfg.MODEL.IMAGE_DECODER.LOSS_KEY = "loss_image_decoder"  # _type: Optional[str]
    # Features from the backbone to use as input
    # potential value(s): ["p2", "p3", "p4", "p5"]
    cfg.MODEL.IMAGE_DECODER.IN_FEATURES = None
    # Outputs from ImageDecoder-FPN heads are up-scaled to the COMMON_STRIDE stride.
    # potential value(s): 4
    cfg.MODEL.IMAGE_DECODER.COMMON_STRIDE = -1
    # Number of channels in the 3x3 convs in the ImageDecoder-FPN heads
    # potential value(s): 32; 64; 128; ...; (keep n*32 if using scale heads group norm)
    cfg.MODEL.IMAGE_DECODER.SCALE_HEADS_DIM = -1
    # Normalization method for the scale heads convolution layers
    # potential value(s): "" (no norm); "GN" (group norm)
    cfg.MODEL.IMAGE_DECODER.SCALE_HEADS_NORM = ""
    # Depth of the pixel predictor (must be >= 1)
    # potential value(s): 1 (no extra layers);  n > 1 (extra layers)
    cfg.MODEL.IMAGE_DECODER.PREDICTOR_DEPTH = 1
    # Number of channels in the 3x3 convs of the extra (non-final) predictor layers.
    # Does nothing for PREDICTOR_DIM == 1
    # potential value(s): the same value as SCALE_HEADS_CONVS_DIM is an easy choice
    cfg.MODEL.IMAGE_DECODER.PREDICTOR_DIM = -1
    # Normalization method for the predictor layers
    # potential value(s): "" (no norm); "GN" (group norm)
    cfg.MODEL.IMAGE_DECODER.PREDICTOR_NORM = ""

    return cfg

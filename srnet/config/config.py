from detectron2.config import CfgNode


def add_srnet_config(cfg: "CfgNode") -> "CfgNode":
    r"""
    Adds the configuration options related to srnets to a `CfgNode` (in-place)
    and returns it.

    Args:
        cfg (detectron2.config.CfgNode): The standard detectron2 config.

    Returns:
        cfg (detectron2.config.CfgNode): The modified config ready for srnets.
    """

    # input aug configuration

    cfg.INPUT.AUG = CfgNode()
    cfg.INPUT.AUG.CUSTOM = CfgNode()
    cfg.INPUT.AUG.CUSTOM.TRAIN = CfgNode()
    cfg.INPUT.AUG.CUSTOM.TRAIN.ENABLED = False
    cfg.INPUT.AUG.CUSTOM.TRAIN.EXTRAS = "last"
    # SEQUENCE structure: [{"NAME": n, "ARGS": {}, "PROB": None}, ...]
    cfg.INPUT.AUG.CUSTOM.TRAIN.SEQUENCE = []
    cfg.INPUT.AUG.CUSTOM.TEST = CfgNode()
    cfg.INPUT.AUG.CUSTOM.TEST.ENABLED = False
    cfg.INPUT.AUG.CUSTOM.TEST.EXTRAS = "last"
    # SEQUENCE structure: [{"NAME": n, "ARGS": {}, "PROB": None}, ...]
    cfg.INPUT.AUG.CUSTOM.TEST.SEQUENCE = []

    # backbones configuration

    cfg.MODEL.FULLY_CONNECTED_BACKBONE = CfgNode()
    cfg.MODEL.FULLY_CONNECTED_BACKBONE.INPUT_SIZE = -1
    cfg.MODEL.FULLY_CONNECTED_BACKBONE.OUT_FEATURES = []
    cfg.MODEL.FULLY_CONNECTED_BACKBONE.LAYER_SIZES = []
    cfg.MODEL.FULLY_CONNECTED_BACKBONE.LAYER_NORMS = []
    cfg.MODEL.FULLY_CONNECTED_BACKBONE.LAYER_ACTIVATIONS = []

    cfg.MODEL.BLOCK_SEQUENCE_BACKBONE = CfgNode()
    cfg.MODEL.BLOCK_SEQUENCE_BACKBONE.OUT_FEATURES = []
    cfg.MODEL.BLOCK_SEQUENCE_BACKBONE.BLOCK_SPECIFICATIONS = [("", {})]

    # image classifier heads configuration

    cfg.MODEL.CLASSIFIER_HEAD = CfgNode()
    cfg.MODEL.CLASSIFIER_HEAD.NAME = "PoolingClassifierHead"
    cfg.MODEL.CLASSIFIER_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.CLASSIFIER_HEAD.LOSS_KEY = "loss_classifier"
    # The number of output classes for the classifier.
    # potential value(s): num_classes > 1
    cfg.MODEL.CLASSIFIER_HEAD.NUM_CLASSES = -1
    # The feature map(s) from the backbone to provide to the classifier head.
    # For PoolingClassifierHead and LinearClassifierHead, this should be a
    # single string, like ["res5"]. Other types of classifier heads may take
    # multiple strings as input.
    cfg.MODEL.CLASSIFIER_HEAD.IN_FEATURES = []  # _type: List[str]
    # For PoolingClassifierHead, the type of global pooling to use.
    # potential value(s): "avg" (global avg pooling); "max" (global max pooling)
    cfg.MODEL.CLASSIFIER_HEAD.POOL_METHOD = "avg"

    # unsupervised learning configuration

    cfg.MODEL.UNSUPERVISED_OBJECTIVE = CfgNode()
    cfg.MODEL.UNSUPERVISED_OBJECTIVE.NAME = ""

    cfg.MODEL.UNSUPERVISED_MULTI_OBJECTIVE = CfgNode()
    # a list of Tuple[str, str, str] representing:
    # (head_name_1, objective_type_1, node_path_1)
    cfg.MODEL.UNSUPERVISED_MULTI_OBJECTIVE.OBJECTIVES_LIST = []
    # an optional space to store configuration for
    cfg.MODEL.UNSUPERVISED_MULTI_OBJECTIVE.OBJECTIVES = CfgNode(new_allowed=True)

    cfg.MODEL.IMAGE_DECODER = CfgNode()
    cfg.MODEL.IMAGE_DECODER.LOSS_WEIGHT = 1.0
    cfg.MODEL.IMAGE_DECODER.LOSS_KEY = "loss_image_decoder"  # _type: Optional[str]
    cfg.MODEL.IMAGE_DECODER.OUTPUT_KEY = "decoded_image"
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

    # For FullyConnectedImageDecoder and BlockSequenceImageDecoder, the output
    # height/width of the image.
    # Note that the output number of channels is derived as len(cfg.MODEL.PIXEL_MEAN)
    # Note also that the final output layer's size must equal output C*H*W.
    cfg.MODEL.IMAGE_DECODER.OUTPUT_HEIGHT = -1
    cfg.MODEL.IMAGE_DECODER.OUTPUT_WIDTH = -1
    # for BlockSequenceImageDecoder, the block sequence configuration.
    # This is a list of tuples of the form (block class, {block init kwargs}).
    cfg.MODEL.IMAGE_DECODER.BLOCK_SPECIFICATIONS = [("", {})]
    # for FullyConnectedImageDecoder, the number of hidden units in each layer
    # of the image decoding network.
    # potential value(s): [n0>=1, n1>=1, ..., nX>=1]
    # Note that the final output layer's size (nX) must equal output C*H*W.
    cfg.MODEL.IMAGE_DECODER.LAYER_SIZES = []
    # for FullyConnectedImageDecoder, the normalization to use in each layer
    # of the image decoding network.
    # potential value(s): ["none" | "bn", ..., normX]
    cfg.MODEL.IMAGE_DECODER.LAYER_NORMS = []
    # for FullyConnectedImageDecoder, the activation function in each layer
    # of the image decoding network.
    # potential value(s): ["relu" | "sigmoid" | "tanh" | "identity", ..., actX]
    cfg.MODEL.IMAGE_DECODER.LAYER_ACTIVATIONS = []

    cfg.MODEL.ORIENTATION_CLASSIFIER = CfgNode()
    cfg.MODEL.ORIENTATION_CLASSIFIER.HEAD_NODE = (
        "MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD"
    )
    cfg.MODEL.ORIENTATION_CLASSIFIER.ANGLE_CLASSES = (
        []
    )  # _type: List[Union[int, float]]
    cfg.MODEL.ORIENTATION_CLASSIFIER.ANGLE_TRANSFORM = ""  # _type: str
    cfg.MODEL.ORIENTATION_CLASSIFIER.LOSS_WEIGHT = 1.0
    cfg.MODEL.ORIENTATION_CLASSIFIER.LOSS_KEY = "loss_orientation_classifier"
    cfg.MODEL.ORIENTATION_CLASSIFIER.OUTPUT_KEY = "pred_orientation_class"
    cfg.MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD = CfgNode(new_allowed=True)
    cfg.MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD.NAME = "PoolingClassifierHead"
    # the following nodes are here so a classifier head can pull them out in
    # its from_config(), but will always be overwritten by the orientation
    # classifier's from_config() with appropriate values from above.
    cfg.MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD.NUM_CLASSES = 2  # unused
    cfg.MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD.LOSS_WEIGHT = 1.0  # unused
    cfg.MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD.LOSS_KEY = (
        "loss_you_shouldnt_see_this"  # unused
    )

    # other args that will be needed by the orientation classifier head should
    # also be placed in the model config's MODEL.ORIENTATION_CLASSIFIER.CLASSIFIER_HEAD
    # node.
    return cfg

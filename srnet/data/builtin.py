from srnet.utils._utils import get_datasets_path

from .datasets.mnist import register_mnist

register_mnist(get_datasets_path())

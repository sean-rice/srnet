from srnet.utils._utils import get_datasets_path

from .datasets.mnist import register_mnist

register_mnist(get_datasets_path())
for prop in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    for seed in range(0, 10 + 1):
        register_mnist(get_datasets_path(), proportion=prop, seed=seed)
for size in range(8, 28 + 2, 2):
    register_mnist(get_datasets_path(), size=size)

from setuptools import find_packages, setup

setup(
    name="srnet",
    version="0.1.0",
    description="sean rice's neural network research",
    packages=find_packages(exclude=("configs",)),
    python_requires=">=3.9",
    install_requires=["fvcore>=0.1.5,<0.1.6", "numpy", "torch",],
)

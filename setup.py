from setuptools import find_packages, setup

setup(
    name="srnet",
    version="0.0.1",
    description="sean rice's neural network research",
    packages=find_packages(exclude=("configs",)),
    python_requires=">=3.8",
    install_requires=["fvcore>=0.1.3,<0.1.4", "numpy", "torch",],
)

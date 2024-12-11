#!/usr/bin/env python
from pathlib import Path
import setuptools

setuptools.setup(
    name="spatial_compare",
    packages=setuptools.find_packages(),
    install_requires=[
        "anndata",
        "jupyter",
        "seaborn",
        "scanpy",
        "scipy",
        "igraph",
        "leidenalg",
        "plotly",
        "kaleido",
    ],
    include_package_data=True,
    version="0.2",
    python_requires=">=3.9",
)

# -*- coding: utf-8 -*-
import os
from shutil import copytree

from setuptools import setup

packages = ["lightweight_openpose", "oak_inference_utils"]

package_data = {
    "": ["*"],
}

extra_requirements = "vis-windows" if os.name == "nt" else "vis"

setup_kwargs = {
    "name": "lightweight_openpose",
    "version": "0.2.0",
    "description": "'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. "
    "It also provides several variants that have some changes to the network structure "
    "for real-time processing on the CPU or low-power embedded devices."
    "The performance of Tensorflow Lite model was tuned significantly",
    "long_description": None,
    "author": "",
    "author_email": "",
    "maintainer": "Xperience.ai",
    "maintainer_email": "hello@xperience.ai",
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.7,<4.0",
    "install_requires": [
        "depthai==0.0.2.1+87247bfb645027a30c68191d88fe1b69b70e39ac",
        f"modelplace-api[{extra_requirements}]@https://github.com/opencv-ai/modelplace-api/archive/v0.4.7.zip",
    ],
}

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(file_dir)
dist_dir = os.path.join(file_dir, "oak_inference_utils")
if not os.path.exists(dist_dir):
    copytree(os.path.join(root_dir, "oak_inference_utils"), dist_dir)

setup(**setup_kwargs)

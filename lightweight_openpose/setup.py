# -*- coding: utf-8 -*-
import os.path as osp

from setuptools import setup

packages = ["lightweight_openpose", "oak_inference_utils"]

package_data = {
    "": ["*"],
}

package_dir = {
    "oak_inference_utils": osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))), "oak_inference_utils",
    ),
}
extra_requirements = "vis-windows"

setup_kwargs = {
    "name": "lightweight_openpose",
    "version": "0.2.2",
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
    "package_dir": package_dir,
    "python_requires": ">=3.7,<4.0",
    "install_requires": [
        "depthai~=2.4.0",
        f"modelplace-api[{extra_requirements}]@https://github.com/opencv-ai/modelplace-api/archive/v0.4.10.zip",
    ],
}

setup(**setup_kwargs)

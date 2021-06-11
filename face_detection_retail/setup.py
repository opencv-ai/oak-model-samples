# -*- coding: utf-8 -*-
import os.path as osp

from setuptools import setup

packages = ["face_detection_retail", "oak_inference_utils"]

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
    "name": "face_detection_retail",
    "version": "0.2.2",
    "description": "Face detector based on SqueezeNet light (half-channels) as a backbone with a single SSD for indoor/outdoor scenes.",
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
        f"modelplace-api[{extra_requirements}]@https://github.com/opencv-ai/modelplace-api/archive/v0.4.10.zip",
        "depthai~=2.4.0",
    ],
}

setup(**setup_kwargs)

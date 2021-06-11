# -*- coding: utf-8 -*-
import os.path as osp

from setuptools import setup

packages = ["textboxes_plus_plus", "oak_inference_utils"]

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
    "name": "textboxes_plus_plus",
    "version": "0.2.2",
    "description": "TextBoxes++ extends TextBoxes for arbitrary oriented text by predicting horizontal bounding boxes as well as quadrilaterals and oriented bounding boxes",
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

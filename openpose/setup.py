# -*- coding: utf-8 -*-
import os.path as osp

from setuptools import setup

packages = ["openpose", "oak_inference_utils"]

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
    "name": "openpose",
    "version": "0.2.2",
    "description": "OpenPose is an industry standard in human body keypoint detection."
    " Given an image of multiple people, the algorithm finds the "
    "keypoints like nose, elbows and knees (17 in total) for everyone. "
    "Then the model builds a skeleton-like representation for every "
    "person on the image. The model was trained on large [MS COCO] "
    "dataset, and because of this, it generalizes well to a multitude "
    "of real-life usecases like surveillance, security, or sports.",
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

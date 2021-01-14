# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["openpose"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "openpose",
    "version": "0.2.0",
    "description": "OpenPose is an industry standard in human body keypoint detection."
    " Given an image of multiple people, the algorithm finds the "
    "keypoints like nose, elbows and knees (17 in total) for everyone. "
    "Then the model builds a skeleton-like representation for every "
    "person on the image. The model was trained on large [MS COCO] "
    "dataset, and because of this, it generalizes well to a multitude "
    "of real-life usecases like surveillance, security, or sports. "
    "This implementation of OpenPose was done by "
    "[Intel® OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) "
    "team, and it was specifically optimized for fast inference on "
    "Intel platforms",
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
        "modelplace-api@git+https://github.com/opencv-ai/modelplace-api.git@v0.1.0-beta#egg=modelplace-api",
    ],
}


setup(**setup_kwargs)

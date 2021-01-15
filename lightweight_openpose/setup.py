# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["lightweight_openpose"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "lightweight_openpose",
    "version": "0.2.0",
    "description": "'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. "
    "It also provides several variants that have some changes to the network structure "
    "for real-time processing on the CPU or low-power embedded devices."
    "The performace of Tensorflow Lite model was tuned significantly"
    "This implementation of OpenPose was done by "
    "[PINTO_model_zoo](https://github.com/PINTO0309/MobileNetV2-PoseEstimation) ",
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
        "modelplace-api[vis]@git+https://github.com/opencv-ai/modelplace-api.git@v0.3.0#egg=modelplace-api",
        "depthai@git+https://github.com/luxonis/depthai-python@gen2_develop#depthai-0.2.0",
    ],
}


setup(**setup_kwargs)

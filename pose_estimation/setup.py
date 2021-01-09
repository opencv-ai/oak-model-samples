# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["pose_estimation"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "pose_estimation",
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
        "modelplace-api@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
    ],
}


setup(**setup_kwargs)

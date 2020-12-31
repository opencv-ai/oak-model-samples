# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["pedestrian_detection_adas"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "pedestrian_detection_adas",
    "version": "0.2.0",
    "description": "Pedestrian detection network based on SSD framework with tuned MobileNet v1 as a feature extractor.",
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
        "modelplace-api[vis]@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
        "depthai@git+https://github.com/luxonis/depthai-python@gen2_develop#depthai-0.2.0"
    ],
}


setup(**setup_kwargs)

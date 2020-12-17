# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["vehicle_detection_adas"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "vehicle_detection_adas",
    "version": "0.2.0",
    "description": "This is a vehicle detection network based on an SSD framework with tuned MobileNet v1 as a feature extractor.",
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

# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["vehicle_license_plate_detection_barrier"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "vehicle_license_plate_detection_barrier",
    "version": "0.2.0",
    "description": "This is a MobileNetV2 + SSD-based vehicle and (Chinese) license plate detector for the 'Barrier' use case.",
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

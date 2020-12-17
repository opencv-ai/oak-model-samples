# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["person_detection_retail"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "person_detection_retail",
    "version": "0.2.0",
    "description": "This is a pedestrian detector for the Retail scenario",
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

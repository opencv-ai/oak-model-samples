# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["tiny_yolo_v3"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "tiny_yolo_v3",
    "version": "0.2.0",
    "description": "YOLO v3 Tiny is a real-time object detection ",
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

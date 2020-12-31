# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["face_detection_adas"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "face_detection_adas",
    "version": "0.2.0",
    "description": "Face detector for driver monitoring and similar scenarios",
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
        "depthai@git+https://github.com/luxonis/depthai-python@gen2_develop#depthai-0.2.0",
        "modelplace-api[vis]@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
    ],
}


setup(**setup_kwargs)

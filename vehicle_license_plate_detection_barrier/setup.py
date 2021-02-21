# -*- coding: utf-8 -*-
import os
from shutil import copytree

from setuptools import setup

packages = ["vehicle_license_plate_detection_barrier", "oak_inference_utils"]

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
        "depthai==0.0.2.1+87247bfb645027a30c68191d88fe1b69b70e39ac",
        "modelplace-api[vis]@https://github.com/opencv-ai/modelplace-api/archive/v0.4.2.zip",
    ],
}

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(file_dir)
dist_dir = os.path.join(file_dir, "oak_inference_utils")
if not os.path.exists(dist_dir):
    copytree(os.path.join(root_dir, "oak_inference_utils"), dist_dir)

setup(**setup_kwargs)

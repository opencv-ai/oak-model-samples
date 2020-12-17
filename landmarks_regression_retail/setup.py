# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["landmarks_regression_retail"]

package_data = {
    "": ["*"],
}

install_requires = [
    "modelplace-api@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
]

setup_kwargs = {
    "name": "landmarks_regression_retail",
    "version": "0.2.0",
    "description": "This is a lightweight landmarks regressor for the Smart Classroom scenario.",
    "long_description": None,
    "author": "",
    "author_email": "",
    "maintainer": "Xperience.ai",
    "maintainer_email": "hello@xperience.ai",
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<4.0",
}


setup(**setup_kwargs)

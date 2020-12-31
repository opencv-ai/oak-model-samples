# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["dbface"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "dbface",
    "version": "0.2.0",
    "description": "",
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
        "depthai==0.0.2.1+22ad34c8264fc3a9a919dbc5c01e3ed3eb41f5aa",
        "modelplace-api@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
    ],
}


setup(**setup_kwargs)

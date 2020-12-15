# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["age_gender_recognition_retail"]

package_data = {
    "": ["*"],
}

install_requires = [
    "depthai==0.0.2.1+22ad34c8264fc3a9a919dbc5c01e3ed3eb41f5aa",
    "model-api @ git+https://github.com/opencv-ai/model-api.git@ps/public-api#egg=model-api-0.2.0",
]

setup_kwargs = {
    "name": "age_gender_recognition_retail",
    "version": "0.2.0",
    "description": "Fully convolutional network for simultaneous Age/Gender recognition.",
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

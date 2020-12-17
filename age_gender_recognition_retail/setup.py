# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["age_gender_recognition_retail"]

package_data = {
    "": ["*"],
}

install_requires = [
    # "depthai==0.0.2.1+7f0633a6f04e5b0c925d372ddc025ddab3879614",
    "modelplace-api@git+https://github.com/opencv-ai/modelplace-api.git#egg=modelplace-api-0.2.0",
]
# depthai@git+https://github.com/luxonis/depthai-python@gen2_develop#depthai-0.2.0
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

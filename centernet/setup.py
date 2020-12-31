# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["centernet"]

package_data = {
    "": ["*"],
}

setup_kwargs = {
    "name": "centernet",
    "version": "0.2.0",
    "description": "CenterNet is a modern object detection model that combines high "
    "quality with elegant and efficient architecture. The idea behind "
    "this algorithm is that it finds the centers of the objects on the "
    "image, and predicts the widths and heights for each of them. This "
    "allows to avoid the usage of so called anchors - pre-set sizes of "
    "the target objects for the network to look for - which means that "
    "CenterNet is a more flexible and efficient approach to detection. "
    "The model was trained on large [MS COCO] dataset, and because of "
    "this, it generalizes well to a multitude of real-life usecases.",
    "long_description": None,
    "author": "Xingyi Zhou",
    "author_email": "zhouxy2017@gmail.com",
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

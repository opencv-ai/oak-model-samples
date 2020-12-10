# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["face_detection_retail"]

package_data = {
    "": ["*"],
}

install_requires = [
    "depthai==0.0.2.1+22ad34c8264fc3a9a919dbc5c01e3ed3eb41f5aa",
    "opencv-python==4.2.0.34",
    "numpy==1.16.4",
    # "model-benchmark-api==0.2.0",
]

dependency_links = [
    "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/",
]

setup_kwargs = {
    "name": "face_detection_retail",
    "version": "0.2.0",
    "description": "Face detector based on SqueezeNet light (half-channels) as a backbone with a single SSD for indoor/outdoor scenes.",
    "long_description": None,
    "author": "",
    "author_email": "",
    "maintainer": "Xperience.ai",
    "maintainer_email": "hello@xperience.ai",
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.7,<4.0",
    "install_requires": install_requires,
    "dependency_links": dependency_links,
}


setup(**setup_kwargs)

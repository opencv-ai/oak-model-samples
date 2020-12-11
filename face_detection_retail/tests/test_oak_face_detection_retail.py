import json
import os

import numpy as np
import pydantic
from model_benchmark_api import Device
from PIL import Image

from face_detection_retail import InferenceModel


def is_equal(result, gt, error=0.001) -> bool:
    if type(result) != type(gt):
        raise TypeError
    ret = True
    if isinstance(result, dict):
        for key in result:
            ret = ret and is_equal(result[key], gt[key], error)
    elif isinstance(result, list):
        for r, g in zip(result, gt):
            ret = ret and is_equal(r, g, error)
    elif isinstance(result, str):
        ret = ret and result == gt
    else:
        ret = ret and np.isclose(result, gt, rtol=error)
    return ret


dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "face_detection.jpg")
test_result_path = os.path.join(dir_name, "face_detection_gt.json")

test_image = Image.open(test_image_path)
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


def test_process_sample_oak_face_detection_retail():
    model = InferenceModel(model_path=model_path)
    model.model_load()
    model.to_device(Device.cpu)
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]

    assert is_equal(ret, test_result["detection"])

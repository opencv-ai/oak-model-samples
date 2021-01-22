import json
import os

import pydantic
from modelplace_api import Device
from modelplace_api.utils import is_equal
from PIL import Image

from palm_detection import InferenceModel
from test_utils import reset_ports

dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "palm_detection.jpg")
test_result_path = os.path.join(dir_name, "palm_detection_gt.json")

test_image = Image.open(test_image_path)
with open(test_result_path) as fp:
    test_result = json.load(fp)


def test_process_sample_palm_detection(reset_ports):
    model = InferenceModel(model_path=model_path, threshold=0.6)
    model.model_load()
    model.to_device(Device.cpu)
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    assert is_equal(ret, test_result)

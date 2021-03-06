import json
import os

import depthai as dai
import pydantic
from modelplace_api.utils import is_equal
from PIL import Image
from retry import retry

from test_utils import reset_ports
from yolov4_tiny import InferenceModel

dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "yolov4_tiny_test.jpg")
test_result_path = os.path.join(dir_name, "yolov4_tiny_gt.json")

test_image = Image.open(test_image_path).convert("RGB")
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


@retry(RuntimeError, tries=3, delay=1)
@reset_ports()
def test_process_sample_tiny_yolov4():
    model = InferenceModel(model_path=model_path, threshold=0.5)
    model.model_load(dai.OpenVINO.VERSION_2020_4)
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    del model
    assert is_equal(ret, test_result)

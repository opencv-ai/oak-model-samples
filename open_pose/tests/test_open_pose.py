import json
import os

import pydantic
from modelplace_api import Device
from modelplace_api.utils import is_equal
from modelplace_api.visualization import draw_pose_estimation_result
from PIL import Image

from open_pose import InferenceModel

dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "000000000139.jpg")
test_result_path = os.path.join(dir_name, "000000000139_gt.json")


test_image = Image.open(test_image_path).convert("RGB")
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


def test_process_sample_open_pose():
    model = InferenceModel(model_path=model_path)
    model.model_load()
    model.to_device(Device.cpu)
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    assert is_equal(ret, test_result["pose_estimation"], 0.02)

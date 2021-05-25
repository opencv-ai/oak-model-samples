import json
import os

import pydantic
from modelplace_api.utils import is_equal
from openvino_hand_pose_estimation import InferenceModel
from PIL import Image

dir_name = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(dir_name)), "test_data")
test_image_path = os.path.join(data_dir, "openvino_hand_detection.jpg")
test_result_path = os.path.join(data_dir, "openvino_hand_detection_gt.json")
test_image = Image.open(test_image_path).convert("RGB")

with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


def test_process_sample_hand_pose_estimation():
    model = InferenceModel(threshold=0.5)
    model.model_load()
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    assert is_equal(ret, test_result)


def test_process_empty_hand_pose_estimation():
    model = InferenceModel()
    model.model_load()
    empty_input = Image.new("RGB", (1280, 720), (128, 128, 128))
    ret = model.process_sample(empty_input)
    assert ret == []

import json
import os

import depthai as dai
import modelplace_api
import pydantic
from modelplace_api.utils import is_equal
from PIL import Image
from retry import retry

from hand_pose_estimation import InferenceModel
from test_utils import reset_ports

dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "hand_pose_estimation.jpg")
test_result_path = os.path.join(dir_name, "hand_pose_estimation_gt.json")

test_image = Image.open(test_image_path).convert("RGB")
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


@retry(RuntimeError, tries=3, delay=1)
@reset_ports()
def test_process_sample_hand_pose_estimation():
    print(modelplace_api.__version__)
    model = InferenceModel(model_path=model_path, threshold=0.3)
    model.model_load(dai.OpenVINO.VERSION_2021_2)
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    del model
    assert is_equal(ret, test_result)

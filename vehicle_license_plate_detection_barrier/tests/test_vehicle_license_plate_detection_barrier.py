import json
import os

import pydantic
from modelplace_api.utils import is_equal
from PIL import Image
from retry import retry

from test_utils import reset_ports
from vehicle_license_plate_detection_barrier import InferenceModel

dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "vehicle_license_plate_detection_barrier.jpeg")
test_result_path = os.path.join(
    dir_name, "vehicle_license_plate_detection_barrier_gt.json",
)

test_image = Image.open(test_image_path)
with open(test_result_path, "r") as j_file:
    test_result = json.loads(j_file.read())


@retry(RuntimeError, tries=3, delay=1)
@reset_ports()
def test_process_sample_person_detection_retail():
    model = InferenceModel(model_path=model_path)
    model.model_load()
    ret = model.process_sample(test_image)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    del model
    assert is_equal(ret, test_result)

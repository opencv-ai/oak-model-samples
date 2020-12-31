import json
import os

import pydantic
from core.utils import is_equal
from modelplace_api import Device
from modelplace_api.visualization import draw_landmarks_result
from PIL import Image

from blaze_face import InferenceModel

# dir_name = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.dirname(dir_name))),
#     "checkpoints",
#     "openvino_blaze_face",
# )
# data_dir = os.path.join(os.path.dirname(os.path.dirname(dir_name)), "test_data")
# test_image_path = os.path.join(dir_name, "kylie-walnut-scrub.jpg")
# # test_result_path = os.path.join(data_dir, "face_detection_gt_openvino.json")
# #
# # with open(test_result_path, "r") as j_file:
# #     test_result = json.loads(j_file.read())


dir_name = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
test_image_path = os.path.join(dir_name, "kylie-walnut-scrub.jpg")
# test_result_path = os.path.join(dir_name, "face_detection_gt.json")
test_image = Image.open(test_image_path).convert("RGB")


def test_process_sample_blaze_face():
    model = InferenceModel(model_path=model_path)
    model.model_load()
    model.to_device(Device.cpu)
    ret = model.process_sample(test_image)
    import cv2

    classes = [str(i) for i in range(6)]
    map_classes = dict([(str(i), [i]) for i in range(5)])
    cv2.imshow(
        "",
        draw_landmarks_result(test_image, ret, classes, map_classes, 0)[-1][:, :, ::-1],
    )
    cv2.waitKey(0)
    ret = [pydantic.json.pydantic_encoder(item) for item in ret]
    # assert is_equal(ret, test_result["detection"])


test_process_sample_blaze_face()

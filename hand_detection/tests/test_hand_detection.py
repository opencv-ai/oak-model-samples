import os

from PIL import Image
from hand_detection import InferenceModel
from modelplace_api import Device
from modelplace_api.visualization import draw_pose_estimation_result

# dir_name = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.join(os.path.dirname(dir_name), "checkpoint")
# test_image_path = os.path.join(dir_name, "facial_landmarks_35_adas.jpg")
# test_result_path = os.path.join(dir_name, "facial_landmarks_35_adas_gt.json")
#
# test_image = Image.open(os.path.join(dir_name, 'LeftHand_2.png')).convert('RGB')


# def test_process_sample_hand_detection():
#     model = InferenceModel(model_path=model_path)
#     model.model_load()
#     model.to_device(Device.cpu)
#     ret = model.process_sample(test_image)
#     import cv2
#     cv2.imshow("", draw_pose_estimation_result(test_image, ret, 0.1)[-1][:, :, ::-1])
#     cv2.waitKey(0)
#
#
# test_process_sample_hand_detection()

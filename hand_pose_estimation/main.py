import os

import depthai as dai
from modelplace_api.visualization import draw_keypoints_one_frame

from hand_pose_estimation import InferenceModel
from oak_inference_utils import inference


def main():
    model_cls = InferenceModel
    root_model_path = os.path.abspath(os.path.dirname(__file__))
    visualization = draw_keypoints_one_frame
    inference(
        model_cls,
        root_model_path,
        visualization,
        openvino_version=dai.OpenVINO.VERSION_2021_2,
    )


if __name__ == "__main__":
    main()

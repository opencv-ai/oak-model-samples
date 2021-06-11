import os

import depthai as dai
from modelplace_api.visualization import draw_landmarks_one_frame

from oak_inference_utils import inference
from palm_detection import InferenceModel


def main():
    model_cls = InferenceModel
    root_model_path = os.path.abspath(os.path.dirname(__file__))
    visualization = draw_landmarks_one_frame
    inference(
        model_cls,
        root_model_path,
        visualization,
        openvino_version=dai.OpenVINO.VERSION_2021_2,
    )


if __name__ == "__main__":
    main()

import os

from modelplace_api.visualization import draw_text_detections_one_frame

from east_text_detector import InferenceModel
from oak_inference_utils import inference


def main():
    model_cls = InferenceModel
    root_model_path = os.path.abspath(os.path.dirname(__file__))
    visualization = draw_text_detections_one_frame
    inference(model_cls, root_model_path, visualization)


if __name__ == "__main__":
    main()

import os

from modelplace_api.visualization import draw_emotion_recognition_one_frame

from emotion_recognition_retail import InferenceModel
from oak_inference_utils import inference


def main():
    root_model_path = os.path.abspath(os.path.dirname(__file__))
    model_cls = InferenceModel
    visualization = draw_emotion_recognition_one_frame
    inference(model_cls, root_model_path, visualization)


if __name__ == "__main__":
    main()

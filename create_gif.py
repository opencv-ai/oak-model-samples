import importlib
import inspect
import os

import cv2
from modelplace_api.visualization import create_gif

from oak_inference_utils.inference import process_video

MODEL_LIST = {
    "landmarks_regression_retail": "draw_landmarks_one_frame",
    "emotion_recognition_retail": "draw_emotion_recognition_one_frame",
    "face_detection_adas": "draw_detections_one_frame",
    "age_gender_recognition_retail": "draw_age_gender_recognition_one_frame",
    "mobilenet_ssd": "draw_detections_one_frame",
    "openpose": "draw_keypints_one_frame",
    "pedestrian_detection_adas": "draw_detections_one_frame",
    "palm_detection": "draw_detections_one_frame",
    "dbface": "draw_landmarks_one_frame",
    "person_detection_retail": "draw_detections_one_frame",
    "vehicle_license_plate_detection_barrier": "draw_detections_one_frame",
    "vehicle_detection_adas": "draw_detections_one_frame",
    "east_text_detector": "draw_text_detections_one_frame",
    "facial_landmarks_35_adas": "draw_landmarks_one_frame",
    "face_detection_retail": "draw_detections_one_frame",
    "lightweight_openpose": "draw_keypints_one_frame",
    "textboxes_plus_plus": "draw_text_detections_one_frame",
    "person_vehicle_bike_detection_crossroad": "draw_detections_one_frame",
    "tiny_yolo_v3": "draw_detections_one_frame",
    "yolov4_tiny": "draw_detections_one_frame",
}


def get_class(class_str: str):
    spec = importlib.util.find_spec(".".join(class_str.split(".")[:-1]))
    class_name = class_str.split(".")[-1]
    if spec is None:
        return None
    else:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    module_classes = [item[0] for item in inspect.getmembers(module, inspect.isclass)]
    module_classes.extend(
        [item[0] for item in inspect.getmembers(module, inspect.isfunction)],
    )
    if class_name in module_classes:
        return getattr(module, class_name)
    else:
        return None


def main():
    root = os.path.dirname(__file__)
    for model, vis_func in MODEL_LIST.items():
        model_path = os.path.join(root, model)
        os.chdir(model_path)
        try:
            model_cls = get_class(f"{model}.InferenceModel")
            assert model_cls is not None
        except AssertionError:
            os.system(
                "python3 setup.py bdist_wheel && rm -R build/ *.egg-info && pip3 install dist/*.whl && rm -R dist/",
            )
        finally:
            model_cls = get_class(f"{model}.InferenceModel")
            vis_func = get_class(f"modelplace_api.visualization.{vis_func}")
            assert model_cls is not None and vis_func is not None
        video_file = os.path.join(model_path, "demo.mp4")
        assert os.path.isfile(video_file)
        model = model_cls(
            model_path=os.path.join(model_path, "checkpoint"), threshold=0.4,
        )
        model.model_load()
        ret = process_video(model, video_file, vis_func)
        cap = cv2.VideoCapture(video_file)
        vis_results = []
        for frame_ret in ret:
            _, frame = cap.read()
            vis_results.append(
                vis_func(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_ret),
            )
        create_gif(
            vis_results,
            video_file.replace(os.path.splitext(video_file)[-1], ".gif"),
            fps=10,
        )
        del model


if __name__ == "__main__":
    main()

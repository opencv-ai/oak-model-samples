import importlib
import inspect
import os
from argparse import ArgumentParser

import cv2
import depthai as dai
import modelplace_api

from oak_inference_utils.inference import process_video


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Name of model to create gif for")
    parser.add_argument(
        "vis_func", type=str, help="Name of visualization function for the model",
    )
    return parser.parse_args()


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
    args = parse_args()
    model_cls = get_class(f"{args.model}.InferenceModel")
    if args.model == "yolov4_tiny":
        openvino_version = dai.OpenVINO.VERSION_2020_4
    elif args.model == "palm_detection" or args.model == "hand_pose_estimation":
        openvino_version = dai.OpenVINO.VERSION_2021_2
    else:
        openvino_version = dai.OpenVINO.VERSION_2020_1
    vis_func = get_class(f"modelplace_api.visualization.{args.vis_func}")
    assert model_cls is not None and vis_func is not None
    video_file = os.path.join(os.getcwd(), "demo.mp4")
    assert os.path.isfile(video_file)
    model = model_cls(
        model_path=os.path.join(os.getcwd(), "checkpoint"), threshold=0.4,
    )
    model.model_load(openvino_version)
    ret = process_video(model, video_file, vis_func)
    cap = cv2.VideoCapture(video_file)
    vis_results = []
    for frame_ret in ret:
        _, frame = cap.read()
        vis_results.append(vis_func(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_ret))
    modelplace_api.visualization.create_gif(
        vis_results,
        video_file.replace(os.path.splitext(video_file)[-1], ".gif"),
        fps=10,
    )
    del model


if __name__ == "__main__":
    main()

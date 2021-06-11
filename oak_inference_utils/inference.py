import json
import os

import cv2
import depthai as dai
import numpy as np
import pydantic
from PIL import Image

from .args_parser import parse_args


def process_frame(image, model, visualization_func):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret = model.process_sample(Image.fromarray(image))
    proceed = True
    if visualization_func is not None:
        vis_result = visualization_func(image, ret)
        cv2.imshow("Visualization", vis_result[:, :, ::-1])
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            proceed = False
    return ret, proceed


def process_video(model, video_file, visualization_func):
    inference_results = []
    cap = cv2.VideoCapture(video_file)
    proceed = True
    while cap.isOpened() and proceed:
        read_correctly, image = cap.read()
        if not read_correctly:
            cv2.destroyAllWindows()
            break
        ret, proceed = process_frame(image, model, visualization_func)
        inference_results.append(ret)
    return inference_results


def process_cam(model, visualization_func):
    inference_results = []
    proceed = True
    while proceed:
        image = model.get_frame_from_camera()
        ret, proceed = process_frame(image, model, visualization_func)
        inference_results.append(ret)
    return inference_results


def inference(
    model_cls,
    root_model_path,
    visualization_func,
    openvino_version=dai.OpenVINO.VERSION_2020_1,
):
    args = parse_args()
    model_path = os.path.join(root_model_path, "checkpoint")
    model = model_cls(
        model_path=model_path,
        threshold=args.threshold,
        preview_shape=args.preview_shape,
    )
    model.model_load(openvino_version=openvino_version, use_camera=args.camera)
    if args.video:
        if not args.visualization:
            visualization_func = None
        inference_results = process_video(model, args.video, visualization_func)
    elif args.camera:
        inference_results = process_cam(model, visualization_func)
    else:
        raise RuntimeError("Choose between video/camera mode")
    if args.save_results:
        with open(os.path.join(root_model_path, "inference_results.json"), "w") as fp:
            json.dump(
                [
                    [pydantic.json.pydantic_encoder(item) for item in frame_result]
                    for frame_result in inference_results
                ],
                fp,
                indent=4,
                sort_keys=True,
            )

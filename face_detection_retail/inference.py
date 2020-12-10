import json
import os
from argparse import ArgumentParser

import cv2
import imageio
import pydantic
from model_benchmark_api import Device
from visualization import draw_detection_result

from face_detection_retail import InferenceModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-cam",
        "--camera",
        action="store_true",
        help="Use DepthAI RGB camera for inference (conflicts with -vid)",
        default=True,
    )
    parser.add_argument(
        "-vid",
        "--video",
        type=str,
        help="Path to file to run inference on (conflicts with -cam)",
    )
    parser.add_argument(
        "-vis",
        "--visualization",
        action="store_true",
        help="Visualize the results from the network (required for -cam)",
    )
    return parser.parse_args()


def inference():
    args = parse_args()
    dir_name = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(dir_name, "checkpoint")
    model = InferenceModel(model_path=model_path)
    model.model_load()
    model.to_device(Device.cpu)
    inference_results = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            read_correctly, image = cap.read()
            if not read_correctly:
                cv2.destroyAllWindows()
                break
            ret = model.process_sample(image)
            inference_results.append(ret)
            if args.visualization:
                vis_result = draw_detection_result(image, ret)
                cv2.imshow("Visualization", vis_result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
    else:
        cam_out = model.add_cam_to_pipeline()
        while True:
            image = cam_out.get().getData().reshape((3, 300, 300)).transpose(1, 2, 0)
            ret = model.process_sample(image)
            inference_results.append(ret)
            if args.visualization:
                vis_result = draw_detection_result(image, ret)
                cv2.imshow("Visualization", vis_result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
            else:
                raise RuntimeError("Camera inference should be used with -vis option")
    with open("inference_results.json", "w") as fp:
        json.dump(
            [
                [pydantic.json.pydantic_encoder(item) for item in frame_result]
                for frame_result in inference_results
            ],
            fp,
            indent=4,
            sort_keys=True,
        )


if __name__ == "__main__":
    inference()

import json
import os
from argparse import ArgumentParser
from os import path as osp

import cv2
import pydantic
from modelplace_api.visualization import create_gif
from visualization import draw_age_gender_recognition_result

from age_gender_recognition_retail import InferenceModel


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
    parser.add_argument(
        "-gif", "--gif", action="store_true", help="Create gif of result visualisation",
    )  # DEBUG, WILL BE REMOVED
    parser.add_argument(
        "-cs",
        "--capture-size",
        help="Frame shapes to capture with DepthAI RGB camera in WxH format."
        " The preview window will have the same shapes (excluding legend).",
        choices=["300x300", "640x480", "1280x720", "1920x1080"],
        default="300x300",
        type=str,
        metavar=("WIDTHxHEIGHT"),
    )
    return parser.parse_args()


def inference():
    args = parse_args()
    dir_name = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(dir_name, "checkpoint")
    model = InferenceModel(model_path=model_path)
    model.model_load()
    inference_results = []
    vis_results = []
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
                vis_result = draw_age_gender_recognition_result(image, ret)
                # DEBUG, WILL BE REMOVED
                if args.gif:
                    vis_results.append(vis_result[..., ::-1])
                cv2.imshow("Visualization", vis_result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
    else:
        preview_width, preview_height = map(int, args.capture_size.split("x"))
        cam_out = model.add_cam_to_pipeline(preview_width, preview_height)
        while True:
            image = (
                cam_out.get()
                .getData()
                .reshape((3, preview_height, preview_width))
                .transpose(1, 2, 0)
            )
            ret = model.process_sample(image)
            inference_results.append(ret)
            if args.visualization:
                vis_result = draw_age_gender_recognition_result(image, ret)
                # DEBUG, WILL BE REMOVED
                if args.gif:
                    vis_results.append(vis_result[..., ::-1])
                cv2.imshow("Visualization", vis_result)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
            else:
                raise RuntimeError("Camera inference should be used with -vis option")

    with open(
        os.path.join(os.path.dirname(__file__), "inference_results.json"), "w",
    ) as fp:
        json.dump(
            [
                [pydantic.json.pydantic_encoder(item) for item in frame_result]
                for frame_result in inference_results
            ],
            fp,
            indent=4,
            sort_keys=True,
        )
    # DEBUG, WILL BE REMOVED
    if args.gif:
        save_path = args.video.replace(osp.splitext(args.video)[-1], ".gif")
        create_gif(vis_results, save_path, fps=10)


if __name__ == "__main__":
    inference()

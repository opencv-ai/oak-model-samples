import importlib
import inspect
from argparse import ArgumentParser
from os import path as osp

import cv2
from modelplace_api.visualization import create_gif
from PIL import Image


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
    parser = ArgumentParser()
    parser.add_argument(
        "-model",
        "-m",
        help="Model to create gif for",
        type=str,
        metavar=("MODELNAME"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="Path to model checkpoint. We assume MODELNAME/checkpoint by default",
        default="checkpoint",
        type=str,
    )
    parser.add_argument(
        "-vis_func",
        "-vf",
        help="Class for model visualization",
        type=str,
        metavar=("VISCLASS"),
        required=True,
    )
    parser.add_argument(
        "-video",
        "-v",
        help="Path to video file for creating gif",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        "-s",
        default=None,
        help="Save path for the created gif. By default we save in the video folder",
        type=str,
    )
    parser.add_argument(
        "--threshold",
        "-tr",
        help="Threshold for model prediction",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--visualization_threshold",
        "-vis_tr",
        help="Threshold for inference result visualization",
        default=0.5,
        type=float,
    )
    args = parser.parse_args()
    class_definition = get_class(args.model + ".InferenceModel")
    visualization = get_class(args.vis_func)
    if class_definition is None or visualization is None:
        raise ImportError
    model = class_definition(
        osp.join(args.model, args.checkpoint), threshold=args.threshold,
    )
    kwargs = {}
    if args.model == "facial_landmarks_35_adas":
        classes = [
            "Left Eye",
            "Right Eye",
            "Nose",
            "Mouth",
            "Left Eyebrow",
            "Right Eyebrow",
            "Face Contour",
        ]
        mapping_classes_to_points = {
            classes[0]: [0, 1],
            classes[1]: [2, 3],
            classes[2]: [4, 5, 6, 7],
            classes[3]: [8, 9, 10, 11],
            classes[4]: [12, 13, 14],
            classes[5]: [15, 16, 17],
            classes[6]: [x for x in range(18, 35)],
        }
        kwargs = {
            "classes": classes,
            "mapping_classes_to_points": mapping_classes_to_points,
            "confidence_threshold": args.visualization_threshold,
        }
    elif args.model == "landmarks_regression_retail" or args.model == "dbface":
        classes = [
            "Left Eye",
            "Right Eye",
            "Nose",
            "Left Lip Corner",
            "Right Lip Corner",
        ]
        mapping_classes_to_points = {
            classes[0]: [0],
            classes[1]: [1],
            classes[2]: [2],
            classes[3]: [3],
            classes[4]: [4],
        }
        kwargs = {
            "classes": classes,
            "mapping_classes_to_points": mapping_classes_to_points,
            "confidence_threshold": args.visualization_threshold,
        }
    elif args.model == "lightweight_openpose":
        kwargs = {
            "confidence_threshold": args.visualization_threshold,
        }

    model.model_load()
    cap = cv2.VideoCapture(args.video)
    vis_results = []
    while cap.isOpened():
        read_correctly, image = cap.read()
        if not read_correctly:
            cv2.destroyAllWindows()
            break
        ret = model.process_sample(Image.fromarray(image[..., ::-1]))
        vis_result = visualization(image, ret, **kwargs)[-1]
        cv2.imshow("Visualization", vis_result)
        vis_results.append(vis_result[..., ::-1])
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
    if args.save_path is None:
        save_path = args.video.replace(osp.splitext(args.video)[-1], ".gif")
    else:
        save_path = args.save_path
    create_gif(vis_results, save_path, fps=10)


if __name__ == "__main__":
    main()

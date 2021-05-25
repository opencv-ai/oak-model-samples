from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-cam",
        "--camera",
        action="store_true",
        help="Use DepthAI RGB camera for inference (conflicts with -vid)",
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
        help="Visualize the results from the network (always on for camera)",
    )
    parser.add_argument(
        "--threshold",
        "-tr",
        help="Threshold for model predictions",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--save-results",
        "-sr",
        help="Save by-frame results of the inference into json",
        action="store_true",
    )
    return parser.parse_args()

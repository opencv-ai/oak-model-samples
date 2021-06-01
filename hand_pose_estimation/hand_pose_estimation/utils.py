from math import atan2, cos, floor, pi, sin
from typing import List

import cv2
import numpy as np
import pydantic
from modelplace_api import Landmarks

SCALE_X = 2.6
SCALE_Y = 2.6
SHIFT_X = 0
SHIFT_Y = -0.5


class RelativePoint(pydantic.BaseModel):
    """
    Point with relative coordinates
    """

    x: float
    y: float


class RelativeBBox(pydantic.BaseModel):
    """
    BBox with relative coordinates
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str


class RelativePalmLabel(pydantic.BaseModel):
    """
    PalmLabel with relative coordinates
    """

    bbox: RelativeBBox
    keypoints: List[RelativePoint]


class HandRegion:
    def __init__(self, pd_score: float, pd_box: list, pd_kps: list):
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps


def normalize_radians(angle: float):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def convert_palm_label_to_relative_coordinates(
    palm_label: Landmarks, height: int, width: int,
):
    """
            Parameters:
                    palm_label (class): Palm Detection result
                    height (int): Input image height
                    width (int): Input image width

            Returns:
                    palm_label (class): palm label with relative coordinates
    """
    relative_coordinates_label = RelativePalmLabel(
        bbox=RelativeBBox(
            x1=palm_label.bbox.x1 / width,
            x2=palm_label.bbox.x2 / width,
            y1=palm_label.bbox.y1 / height,
            y2=palm_label.bbox.y2 / height,
            score=palm_label.bbox.score,
            class_name=palm_label.bbox.class_name,
        ),
        keypoints=[
            RelativePoint(x=keypoint.x / width, y=keypoint.y / height)
            for keypoint in palm_label.keypoints
        ],
    )
    return relative_coordinates_label


def convert_palm_labels_to_hand_regions(
    palm_labels: List[Landmarks], height: int, width: int,
):
    regions = []
    for palm_label in palm_labels:
        palm_label = convert_palm_label_to_relative_coordinates(
            palm_label, height, width,
        )
        palm_w = palm_label.bbox.x2 - palm_label.bbox.x1
        palm_h = palm_label.bbox.y2 - palm_label.bbox.y1
        regions.append(
            HandRegion(
                pd_score=palm_label.bbox.score,
                pd_box=[palm_label.bbox.x1, palm_label.bbox.y1, palm_w, palm_h],
                pd_kps=palm_label.keypoints,
            ),
        )
    return regions


def convert_hand_regions_to_rect(regions: List[HandRegion], w: int, h: int):
    """
        https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
        Converts results of palm detection into a rectangle (normalized by image size)
        that encloses the palm and is rotated such that the line connecting center of
        the wrist and MCP of the middle finger is aligned with the Y-axis of the
    """

    target_angle = pi * 0.5  # 90 = pi/2
    for region in regions:
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0].x, region.pd_kps[0].y  # wrist center
        x1, y1 = region.pd_kps[2].x, region.pd_kps[2].y  # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)

    # shift rectangle
    rect_transformation(regions, w, h)


def rotated_rect_to_points(cx: float, cy: float, w: float, h: float, rotation: float):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0x = cx - a * h - b * w
    p0y = cy + b * h - a * w
    p1x = cx + a * h - b * w
    p1y = cy - b * h - a * w
    p2x = int(2 * cx - p0x)
    p2y = int(2 * cy - p0y)
    p3x = int(2 * cx - p1x)
    p3y = int(2 * cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]


def rect_transformation(regions: List[HandRegion], w: int, h: int):
    """
        Parameters:
            regions (List[HandRegion]): list with hand regions
            height (int): Input image height
            width (int): Input image width
        https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
        Expands and shifts the rectangle that contains the palm so that it's likely to cover the entire hand.
    """
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * SHIFT_X) * w
            region.rect_y_center_a = (region.rect_y_center + height * SHIFT_Y) * h
        else:
            x_shift = w * width * SHIFT_X * cos(rotation) - h * height * SHIFT_Y * sin(
                rotation,
            )
            y_shift = w * width * SHIFT_X * sin(rotation) + h * height * SHIFT_Y * cos(
                rotation,
            )
            region.rect_x_center_a = region.rect_x_center * w + x_shift
            region.rect_y_center_a = region.rect_y_center * h + y_shift

        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * SCALE_X
        region.rect_h_a = long_side * SCALE_Y
        region.rect_points = rotated_rect_to_points(
            region.rect_x_center_a,
            region.rect_y_center_a,
            region.rect_w_a,
            region.rect_h_a,
            region.rotation,
        )


def warp_rect_img(rect_points: list, img: np.ndarray, w: int, h: int):
    src = np.array(rect_points[1:], dtype=np.float32)
    dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, mat, (w, h))

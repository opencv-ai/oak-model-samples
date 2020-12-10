from collections import defaultdict
from typing import List

import cv2
import numpy as np
from colors import RGB_COLORS
from model_benchmark_api import BBox


def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple,
    thickness: int = 1,
    bg_color: tuple = (128, 128, 128),
    font_scale: float = 0.5,
) -> np.ndarray:
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    baseline += thickness
    text_org = np.array((origin[0], origin[1] - text_size[1]))
    cv2.rectangle(
        img,
        tuple((text_org + (0, baseline)).astype(int)),
        tuple((text_org + (text_size[0], -text_size[1])).astype(int)),
        bg_color,
        -1,
    )

    cv2.putText(
        img,
        text,
        tuple((text_org + (0, baseline / 2)).astype(int)),
        font_face,
        font_scale,
        (0, 0, 0),
        thickness,
        8,
    )

    return img


def draw_legend(
    image: np.ndarray,
    class_map: dict,
    thickness: int = 3,
    cell_height: int = 20,
    height: int = 10,
    weight: int = 10,
) -> np.ndarray:
    bg_color = (255, 255, 255)
    x0 = 0
    y0 = 0
    legend = np.zeros((image.shape[0], 200, 3), dtype=image.dtype)
    legend.fill(255)
    cv2.rectangle(
        legend,
        (y0, x0),
        (x0 + 200, y0 + cell_height * (len(class_map.keys()) + 3) + 5),
        bg_color,
        -1,
    )
    draw_text(legend, "Labels:", (weight, height + 15), bg_color=bg_color)
    height += cell_height
    for class_name, class_color in class_map.items():
        cv2.line(
            legend, (weight + 2, height), (weight + 40, height), class_color, thickness,
        )
        draw_text(legend, class_name, (weight + 45, height + 15), bg_color=bg_color)
        height += cell_height
    return np.concatenate([image, legend], axis=1)


def draw_detection_result(
    image: np.ndarray, detections: List[BBox], thickness: int = 2,
) -> np.ndarray:
    image = image.copy()
    possible_labels = list(set([det.class_name for det in detections]))
    class_map = dict(
        [
            [possible_labels[class_number], RGB_COLORS[class_number][::-1]]
            for class_number in range(len(possible_labels))
        ],
    )

    classes = defaultdict(list)
    for det in detections:
        classes[det.class_name].append(det)
    for class_name, class_detections in classes.items():
        for detection in class_detections:
            label_id = possible_labels.index(detection.class_name)
            color = RGB_COLORS[label_id][::-1]
            image = cv2.rectangle(
                image,
                (int(detection.x1), int(detection.y1)),
                (int(detection.x2), int(detection.y2)),
                tuple(color),
                thickness=thickness,
            )
    return draw_legend(image, class_map)

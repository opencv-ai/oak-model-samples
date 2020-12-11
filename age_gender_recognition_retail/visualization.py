from typing import List

import cv2
import numpy as np
from colors import RGB_COLORS
from model_benchmark_api import AgeGenderLabel


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


def draw_classification_legend(
    image: np.ndarray,
    class_map: dict,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    width_offset: int = 10,
    height_offset: int = 15,
    cell_height: int = 40,
    height: int = 10,
    weight: int = 10,
) -> np.ndarray:
    bg_color = (255, 255, 255)
    x0 = 0
    y0 = 0
    legend = np.zeros((image.shape[0], 300, 3), dtype=image.dtype)
    legend.fill(255)
    cv2.rectangle(
        legend,
        (y0, x0),
        (x0 + 300, y0 + cell_height * (len(class_map.keys()) + 3) + 5),
        bg_color,
        -1,
    )
    draw_text(
        legend,
        "Labels:",
        (weight, height + height_offset),
        bg_color=bg_color,
        font_scale=font_scale,
        thickness=font_thickness,
    )
    height += cell_height
    for class_name, class_score in class_map.items():
        draw_text(
            legend,
            f"{class_name}: {class_score}",
            (weight + width_offset, height),
            bg_color=bg_color,
            font_scale=font_scale,
            thickness=font_thickness,
        )
        height += cell_height
    return np.concatenate([image, legend], axis=1)


def draw_age_gender_recognition_result(
    image: np.ndarray, detections: List[AgeGenderLabel],
) -> List[np.ndarray]:
    image = image.copy()
    for detection in detections:
        image = cv2.rectangle(
            image,
            (int(detection.bbox.x1), int(detection.bbox.y1)),
            (int(detection.bbox.x2), int(detection.bbox.y2)),
            RGB_COLORS[196],
            thickness=8,
        )
        classification_labels = {
            label.class_name.capitalize(): round(float(label.score), 2)
            for label in detection.gender
        }
        classification_labels["Age"] = detection.age
        image = draw_classification_legend(
            image,
            classification_labels,
            font_scale=0.75,
            font_thickness=2,
            height_offset=20,
            cell_height=50,
        )
    return image

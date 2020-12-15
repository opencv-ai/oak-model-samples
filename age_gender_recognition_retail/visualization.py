from typing import List

import cv2
import numpy as np
from model_api import RGB_COLORS, AgeGenderLabel, draw_classification_legend


def draw_age_gender_recognition_result(
    image: np.ndarray, detections: List[AgeGenderLabel],
) -> np.ndarray:
    image = np.ascontiguousarray(image)
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

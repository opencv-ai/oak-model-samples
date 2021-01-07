from typing import List

import cv2
import numpy as np
from modelplace_api import AgeGenderLabel
from modelplace_api.visualization import RGB_COLORS, draw_classification_legend


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
        # THis should be a function in modelplace_api visualization
        bg_color = RGB_COLORS[196]
        font_scale = 1
        font = cv2.FONT_HERSHEY_TRIPLEX
        label_score, label_class_name = detection.gender[0]
        age = detection.age
        text = f'{label_class_name[1]}, Age: {age}'
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        text_offset_x = detection.bbox.x1
        text_offset_y = detection.bbox.y1 - 15
        box_coords = ((int(text_offset_x), int(text_offset_y)),
                      (int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)))
        cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(image, text,
                    (int(text_offset_x), int(text_offset_y)),
                    font, fontScale=font_scale,
                    color=(255, 255, 255), thickness=1)
    return image

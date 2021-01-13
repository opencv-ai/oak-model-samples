from typing import List, Union

import cv2
import numpy as np
from modelplace_api import AgeGenderLabel
from modelplace_api.visualization import RGB_COLORS
from PIL.Image import Image


def draw_text_label(
    image: np.ndarray,
    detection: AgeGenderLabel,
    text: str,
    font_scale: float = 0.6,
    thickness: int = 1,
    bg_color: tuple = (128, 128, 128),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_TRIPLEX
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=thickness,
    )[0]
    text_offset_x = detection.bbox.x1
    text_offset_y = detection.bbox.y1 - 10
    box_coords = (
        (int(text_offset_x), int(text_offset_y)),
        (int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)),
    )
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (int(text_offset_x), int(text_offset_y)),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=1,
    )
    return image


def draw_emotion_recognition_result(
    image: Union[Image, np.ndarray], detections: List[EmotionLabel],
) -> List[np.ndarray]:
    image_with_boxes = np.ascontiguousarray(image)
    source_image = image_with_boxes.copy()
    images = list()
    images.append(source_image)
    for detection in detections:
        image_with_boxes = cv2.rectangle(
            image_with_boxes,
            (int(detection.bbox.x1), int(detection.bbox.y1)),
            (int(detection.bbox.x2), int(detection.bbox.y2)),
            RGB_COLORS[196],
            thickness=8,
        )
        bg_color = RGB_COLORS[196]
        label_score, label_class_name = detection.emotion[0]
        image_with_boxes = draw_text_label(
            image_with_boxes, detection, label_class_name[1], bg_color=bg_color,
        )
        images.append(image_with_boxes)
    return images

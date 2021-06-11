from typing import Tuple

import cv2
import numpy as np
from modelplace_api import Landmarks, Point

from oak_inference_utils import OAKTwoStageModel

from .face_processing import FaceProcessor, pad_img


class InferenceModel(OAKTwoStageModel):
    def __init__(
        self,
        model_path: str,
        preview_shape: Tuple[int, int] = (640, 480),
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        area_threshold: float = 0.15,
        face_bbox_pad_percent: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="data",
            preview_shape=preview_shape,
            first_stage=FaceProcessor(threshold, preview_shape),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.area_threshold = area_threshold
        self.face_bbox_pad_percent = face_bbox_pad_percent
        self.input_height, self.input_width = 48, 48

    def preprocess(self, data):
        face_bboxes = self.get_first_stage_result(data)
        preprocessed_data = []
        preprocessed_bboxes = []
        if face_bboxes == [[]]:
            return None
        for i, img in enumerate(data):
            areas = [
                (bbox.y2 - bbox.y1) * (bbox.x2 - bbox.x1) for bbox in face_bboxes[i]
            ]
            max_area = max(areas)
            preprocessed_img = []
            img_bboxes = []
            img = np.array(img)[:, :, ::-1]
            for j, face_bbox in enumerate(face_bboxes[i]):
                if areas[j] < self.area_threshold * max_area:
                    continue
                # apply padding to face bbox to increase quality of facial landmark model
                face_bbox_width = face_bbox.x2 - face_bbox.x1
                face_bbox_hight = face_bbox.y2 - face_bbox.y1
                face_bbox.x1 = int(
                    np.clip(
                        face_bbox.x1 - face_bbox_width * self.face_bbox_pad_percent,
                        0,
                        img.shape[1],
                    ),
                )
                face_bbox.y2 = int(
                    np.clip(
                        face_bbox.y2 + face_bbox_hight * self.face_bbox_pad_percent,
                        0,
                        img.shape[0],
                    ),
                )
                face_bbox.x2 = int(
                    np.clip(
                        face_bbox.x2 + face_bbox_width * self.face_bbox_pad_percent,
                        0,
                        img.shape[1],
                    ),
                )
                cropped_face = img[
                    face_bbox.y1 : face_bbox.y2, face_bbox.x1 : face_bbox.x2,
                ]
                height, width, _ = cropped_face.shape
                if self.input_height / self.input_width < height / width:
                    scale = self.input_height / height
                else:
                    scale = self.input_width / width

                scaled_img = cv2.resize(
                    cropped_face,
                    (0, 0),
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                padded_img, pad = pad_img(
                    scaled_img, (0, 0, 0), [self.input_height, self.input_width],
                )

                padded_img = padded_img.transpose((2, 0, 1))
                padded_img = padded_img[np.newaxis].astype(np.float32)
                preprocessed_img.append(padded_img)
                img_bboxes.append(face_bbox)
            preprocessed_data.append(preprocessed_img)
            preprocessed_bboxes.append(img_bboxes)

        return [preprocessed_data, preprocessed_bboxes]

    def postprocess(self, predictions):
        postprocessed_result = []

        for results, face_bboxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for result, face_bbox in zip(results, face_bboxes):
                keypoints = np.array(result.getLayerFp16(result.getAllLayerNames()[0]))
                bbox_w, bbox_h = (
                    (face_bbox.x2 - face_bbox.x1),
                    (face_bbox.y2 - face_bbox.y1),
                )
                image_predictions.append(
                    Landmarks(
                        bbox=face_bbox,
                        keypoints=[
                            Point(
                                x=face_bbox.x1 + bbox_w * x,
                                y=face_bbox.y1 + bbox_h * y,
                            )
                            for x, y in zip(keypoints[::2], keypoints[1::2])
                        ],
                    ),
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

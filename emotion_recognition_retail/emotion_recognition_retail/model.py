import cv2
import numpy as np
from modelplace_api import EmotionLabel, Label

from oak_inference_utils import OAKTwoStageModel, pad_img

from .face_processing import FaceProcessor


class InferenceModel(OAKTwoStageModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        area_threshold: float = 0.6,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="data",
            first_stage=FaceProcessor(threshold),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.area_threshold = area_threshold
        self.classes = ["neutral", "happy", "sad", "surprise", "anger"]
        self.input_height, self.input_width = 64, 64

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
                if face_bbox.x2 - face_bbox.x1 < self.input_width:
                    continue

                cropped_face = img[
                    int(face_bbox.y1) : int(face_bbox.y2),
                    int(face_bbox.x1) : int(face_bbox.x2),
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
        for result, face_bboxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for face_bbox, result_emotion_prob in zip(face_bboxes, result):
                emotions_probs_list = result_emotion_prob.getLayerFp16(
                    result_emotion_prob.getAllLayerNames()[0],
                )
                emotions_index = np.argsort(emotions_probs_list)
                image_predictions.append(
                    EmotionLabel(
                        bbox=face_bbox,
                        emotions=[
                            Label(
                                score=emotions_probs_list[idx],
                                class_name=self.classes[idx],
                            )
                            for idx in reversed(emotions_index)
                        ],
                    ),
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

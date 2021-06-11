import cv2
import numpy as np
from modelplace_api import Point, TextPolygon

from oak_inference_utils import DataInfo, OAKSingleStageModel

from .postprocessing import decode_predictions, non_max_suppression, rotated_rectangle


class InferenceModel(OAKSingleStageModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="input_images",
            input_shapes=(320, 320),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold
        self.mean = np.array([123.68, 116.78, 103.94], dtype="float32")

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
            height, width, _ = img.shape
            scale_x = self.input_width / width
            scale_y = self.input_height / height
            scaled_img = cv2.resize(img, (self.input_height, self.input_width))
            scaled_img = scaled_img.astype(np.float32)
            scaled_img -= self.mean
            scaled_img = scaled_img.transpose((2, 0, 1))
            scaled_img = scaled_img[np.newaxis]
            data_infos.append(
                DataInfo(
                    scales=(scale_x, scale_y),
                    pads=(0, 0),
                    original_width=width,
                    original_height=height,
                ),
            )
            preprocessed_data.append(scaled_img)
        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []

        for result, input_info in zip(predictions[0], predictions[1]):
            scores = np.array(
                result.getLayerFp16(result.getAllLayerNames()[0]),
            ).reshape((1, 1, 80, 80))
            geometry1 = np.array(
                result.getLayerFp16(result.getAllLayerNames()[1]),
            ).reshape((1, 4, 80, 80))
            geometry2 = np.array(
                result.getLayerFp16(result.getAllLayerNames()[2]),
            ).reshape((1, 1, 80, 80))
            scale_x, scale_y = input_info.scales
            (boxes, confidences) = decode_predictions(
                scores, geometry1, geometry2, self.threshold, scale_x, scale_y,
            )
            boxes = non_max_suppression(boxes, probs=confidences)

            image_predictions = []
            for box in boxes:
                if len(box) == 0:
                    continue
                original_w = self.input_width / scale_x
                original_h = self.input_height / scale_y
                points = rotated_rectangle(box)
                points = [
                    Point(x=np.clip(x, 0, original_w), y=np.clip(y, 0, original_h))
                    for x, y in points
                ]
                image_predictions.append(TextPolygon(points=points))
            postprocessed_result.append(image_predictions)
        return postprocessed_result

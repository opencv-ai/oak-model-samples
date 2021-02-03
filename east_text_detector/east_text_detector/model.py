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

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
            height, width, _ = img.shape
            scale_x = width / float(self.input_width)
            scale_y = height / float(self.input_height)
            scaled_img = cv2.resize(img, (self.input_height, self.input_width))
            scaled_img = scaled_img.astype(np.float32)
            scaled_img = scaled_img.transpose((2, 0, 1))
            scaled_img = scaled_img[np.newaxis]
            data_infos.append(
                DataInfo(
                    scales=(scale_y, scale_x),
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
            boxes, confidences, angles = decode_predictions(
                scores, geometry1, geometry2, self.threshold,
            )
            boxes, angles = non_max_suppression(
                np.array(boxes), probs=confidences, angles=np.array(angles),
            )
            scale_y, scale_x = input_info.scales
            image_predictions = []
            for box, confidence, angle in zip(boxes, confidences, angles):
                original_w = scale_x * self.input_width
                original_h = scale_y * self.input_height
                x1 = int(np.clip(box[0] * scale_x, 0, original_w))
                y1 = int(np.clip(box[1] * scale_y, 0, original_h))
                x2 = int(np.clip(box[2] * scale_x, 0, original_w))
                y2 = int(np.clip(box[3] * scale_y, 0, original_h))
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                center_x = int(x1 + width / 2)
                center_y = int(y1 + height / 2)
                rotated_rect = ((center_x, center_y), ((x2 - x1), (y2 - y1)), -angle)
                points = rotated_rectangle(rotated_rect)
                points = [Point(x=x, y=y) for x, y in points]
                image_predictions.append(TextPolygon(points=points))
            postprocessed_result.append(image_predictions)
        return postprocessed_result

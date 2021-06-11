import json
import os

import cv2
import numpy as np
from modelplace_api import Point, TextPolygon

from oak_inference_utils import DataInfo, OAKSingleStageModel, pad_img

from .postprocessing import PriorUtil


class InferenceModel(OAKSingleStageModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="input_1",
            input_shapes=(256, 256),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold
        with open(
            os.path.join(os.path.dirname(__file__), "postprocessing_config.json"), "r",
        ) as fp:
            postprocess_config = json.load(fp)
        self.postprocessor = PriorUtil(**postprocess_config)

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
            height, width, _ = img.shape
            if self.input_height / self.input_width < height / width:
                scale = self.input_height / height
            else:
                scale = self.input_width / width

            scaled_img = cv2.resize(
                img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
            )
            padded_img, pad = pad_img(
                scaled_img, (0, 0, 0), [self.input_height, self.input_width],
            )

            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float32)
            preprocessed_data.append(padded_img)
            data_infos.append(
                DataInfo(
                    scales=(scale, scale),
                    pads=tuple(pad),
                    original_width=width,
                    original_height=height,
                ),
            )

        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []

        for result, input_info in zip(predictions[0], predictions[1]):
            (scale_x, scale_y), pads = input_info.scales, input_info.pads
            original_w, original_h = (
                input_info.original_width,
                input_info.original_height,
            )
            h, w = self.input_height, self.input_width
            result = np.concatenate(
                [
                    np.array(result.getLayerFp16(result.getAllLayerNames()[3])).reshape(
                        -1, 5,
                    ),
                    np.array(result.getLayerFp16(result.getAllLayerNames()[2])).reshape(
                        -1, 8,
                    ),
                    np.array(result.getLayerFp16(result.getAllLayerNames()[1])).reshape(
                        -1, 4,
                    ),
                    np.array(result.getLayerFp16(result.getAllLayerNames()[0])).reshape(
                        -1, 2,
                    ),
                ],
                axis=-1,
            )

            quads = self.postprocessor.decode_results(result, self.threshold).reshape(
                -1, 4, 2,
            )
            quads[:, :, 0] = np.clip(
                (quads[:, :, 0] * w - pads[1]) / scale_x, 0, original_w,
            )
            quads[:, :, 1] = np.clip(
                (quads[:, :, 1] * h - pads[0]) / scale_y, 0, original_h,
            )

            image_predictions = []
            for quad in quads:
                points = [Point(x=int(x), y=int(y)) for x, y in quad]
                image_predictions.append(TextPolygon(points=points))
            postprocessed_result.append(image_predictions)

        return postprocessed_result

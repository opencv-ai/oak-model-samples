import os

import cv2
import numpy as np
from modelplace_api import BBox, Landmarks, Point

from oak_inference_utils import DataInfo, OAKSingleStageModel, pad_img

from .postprocessing import Postprocessor


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
            input_name="input",
            input_shapes=(128, 128),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold
        self.output_names = ["classificators", "regressors"]
        self.class_names = {0: "background", 1: "palm"}
        self.postprocessor = Postprocessor(
            os.path.join(os.path.dirname(__file__), "anchors.csv"),
        )

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
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
            image_predictions = []
            result = [
                np.array(result.getLayerFp16(self.output_names[0])),
                np.array(result.getLayerFp16(self.output_names[1])).reshape(-1, 18),
            ]
            if result[0].size == 0:
                postprocessed_result.append(image_predictions)
                return postprocessed_result
            boxes, keypoints = self.postprocessor.decode_predictions(
                result, self.output_names,
            )
            for box, kps in zip(boxes, keypoints):
                if box[4] > self.threshold:
                    palm_box = BBox(
                        x1=int(
                            np.clip((box[0] * w - pads[1]) / scale_x, 0, original_w),
                        ),
                        y1=int(
                            np.clip((box[1] * h - pads[0]) / scale_y, 0, original_h),
                        ),
                        x2=int(
                            np.clip((box[2] * w - pads[1]) / scale_x, 0, original_w),
                        ),
                        y2=int(
                            np.clip((box[3] * h - pads[0]) / scale_y, 0, original_h),
                        ),
                        score=float(box[4]),
                        class_name=self.class_names[1],
                    )
                    image_predictions.append(
                        Landmarks(
                            bbox=palm_box,
                            keypoints=[
                                Point(
                                    x=int((keypoint[0] * w - pads[1]) / scale_x),
                                    y=int((keypoint[1] * h - pads[0]) / scale_y),
                                )
                                for keypoint in kps
                            ],
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

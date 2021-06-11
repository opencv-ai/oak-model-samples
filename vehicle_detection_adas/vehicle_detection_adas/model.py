import cv2
import numpy as np
from modelplace_api import BBox

from oak_inference_utils import DataInfo, OAKSingleStageModel, pad_img


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
            input_name="data",
            input_shapes=(672, 384),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold

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
            boxes = np.array(result.getLayerFp16(result.getAllLayerNames()[0])).reshape(
                -1, 7,
            )
            image_predictions = []
            for box in boxes:
                if box[2] > self.threshold:
                    image_predictions.append(
                        BBox(
                            x1=int(
                                np.clip(
                                    (box[3] * w - pads[1]) / scale_x, 0, original_w,
                                ),
                            ),
                            y1=int(
                                np.clip(
                                    (box[4] * h - pads[0]) / scale_y, 0, original_h,
                                ),
                            ),
                            x2=int(
                                np.clip(
                                    (box[5] * w - pads[1]) / scale_x, 0, original_w,
                                ),
                            ),
                            y2=int(
                                np.clip(
                                    (box[6] * h - pads[0]) / scale_y, 0, original_h,
                                ),
                            ),
                            score=float(box[2]),
                            class_name="vehicle",
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

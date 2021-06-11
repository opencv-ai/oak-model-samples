import cv2
import depthai as dai
import numpy as np
from modelplace_api import BBox

from oak_inference_utils import DataInfo, pad_img, wait_for_results


class FaceProcessor:
    def __init__(self, threshold, preview_shape):
        self.threshold = threshold
        self.input_height, self.input_width = (300, 300)
        self.video_width, self.video_height = preview_shape

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
                            class_name="person",
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def get_input_shapes(self):
        return self.input_width, self.input_height

    def forward(self, in_queue, out_queue, data):
        results = []
        if data is not None:
            for sample in data[0]:
                nn_data = dai.NNData()
                nn_data.setLayer("data", sample)
                in_queue.send(nn_data)
                assert wait_for_results(out_queue)
                results.append(out_queue.get())
            data[0] = results
        else:
            assert wait_for_results(out_queue)
            results.append(out_queue.get())
            data = [
                results,
                [
                    DataInfo(
                        scales=(
                            self.input_width / self.video_width,
                            self.input_height / self.video_height,
                        ),
                        pads=(0, 0),
                        original_width=self.video_width,
                        original_height=self.video_height,
                    ),
                ],
            ]
        return data

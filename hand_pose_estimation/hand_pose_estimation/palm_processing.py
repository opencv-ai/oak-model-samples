import cv2
import depthai as dai
import numpy as np
from modelplace_api import Point, BBox, Landmarks
import os
from oak_inference_utils import DataInfo, pad_img, wait_for_results
from .postprocessing import Postprocessor


class PalmProcessor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.input_height, self.input_width = (128, 128)
        self.output_names = ["classificators", "regressors"]
        self.postprocessor = Postprocessor(
            os.path.join(os.path.dirname(__file__), "anchors.csv"),
        )
        self.class_names = {0: "background", 1: "palm"}

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
            scale, pads = input_info.scales[0], input_info.pads
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
                        x1=int(np.clip((box[0] * w - pads[1]) / scale, 0, original_w)),
                        y1=int(np.clip((box[1] * h - pads[0]) / scale, 0, original_h)),
                        x2=int(np.clip((box[2] * w - pads[1]) / scale, 0, original_w)),
                        y2=int(np.clip((box[3] * h - pads[0]) / scale, 0, original_h)),
                        score=float(box[4]),
                        class_name=self.class_names[1],
                    )
                    image_predictions.append(
                        Landmarks(
                            bbox=palm_box,
                            keypoints=[
                                Point(
                                    x=int((keypoint[0] * w - pads[1]) / scale),
                                    y=int((keypoint[1] * h - pads[0]) / scale),
                                )
                                for keypoint in kps
                            ],
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def get_input_shapes(self):
        return self.input_width, self.input_height

    @staticmethod
    def forward(in_queue, out_queue, data):
        results = []
        for sample in data[0]:
            nn_data = dai.NNData()
            nn_data.setLayer("data", sample)
            in_queue.send(nn_data)
            assert wait_for_results(out_queue)
            results.append(out_queue.get())
        data[0] = results
        return data
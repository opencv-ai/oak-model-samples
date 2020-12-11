import math
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from model_api import BBox


def pad_img(img, pad_value, target_dims):
    h, w, _ = img.shape
    pads = []
    pads.append(int(math.floor((target_dims[0] - h) / 2.0)))
    pads.append(int(math.floor((target_dims[1] - w) / 2.0)))
    pads.append(int(target_dims[0] - h - pads[0]))
    pads.append(int(target_dims[1] - w - pads[1]))
    padded_img = cv2.copyMakeBorder(
        img, pads[0], pads[2], pads[1], pads[3], cv2.BORDER_CONSTANT, value=pad_value,
    )
    return padded_img, pads


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


class FaceProcessor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.class_names = {
            0: "background",
            1: "person",
        }
        self.input_height, self.input_width = (300, 300)

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
            planar_img = padded_img.flatten().astype(np.float32)
            preprocessed_data.append(planar_img)
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []

        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pads = input_info
            h, w = self.input_height, self.input_width
            original_h = int((h - (pads[0] + pads[2])) / scale)
            original_w = int((w - (pads[1] + pads[3])) / scale)
            boxes = np.array(result.getLayerFp16(result.getAllLayerNames()[0])).reshape(
                -1, 7,
            )
            boxes = boxes[boxes[:, 2] > self.threshold]
            image_predictions = []
            for box in boxes:
                image_predictions.append(
                    BBox(
                        x1=float(
                            np.clip((box[3] * w - pads[1]) / scale, 0, original_w),
                        ),
                        y1=float(
                            np.clip((box[4] * h - pads[0]) / scale, 0, original_h),
                        ),
                        x2=float(
                            np.clip((box[5] * w - pads[1]) / scale, 0, original_w),
                        ),
                        y2=float(
                            np.clip((box[6] * h - pads[0]) / scale, 0, original_h),
                        ),
                        score=float(box[2]),
                        class_name=self.class_names[int(box[1])],
                    ),
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def forward(self, in_queue, out_queue, data):
        results = []
        for sample in data[0]:
            nn_data = dai.NNData()
            nn_data.setLayer("data", sample)
            in_queue.send(nn_data)
            assert wait_for_results(out_queue)
            results.append(out_queue.get())
        data[0] = results
        return data

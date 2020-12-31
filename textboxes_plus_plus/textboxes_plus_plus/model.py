import json
import math
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, Point, TextPolygon

from .postprocessing import PriorUtil


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


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


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.input_height, self.input_width = 256, 256

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
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []

        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pads = input_info
            h, w = self.input_height, self.input_width
            original_h = int((h - (pads[0] + pads[2])) / scale)
            original_w = int((w - (pads[1] + pads[3])) / scale)
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
                (quads[:, :, 0] * w - pads[1]) / scale, 0, original_w,
            )
            quads[:, :, 1] = np.clip(
                (quads[:, :, 1] * h - pads[0]) / scale, 0, original_h,
            )
            image_predictions = []
            for quad in quads:
                points = [Point(x=x, y=y) for x, y in quad]
                image_predictions.append(TextPolygon(points=points))
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

    def create_pipeline(self, model_blob):
        self.pipeline = dai.Pipeline()

        data_in = self.pipeline.createXLinkIn()
        data_in.setStreamName("data_in")

        model = self.pipeline.createNeuralNetwork()
        model.setBlobPath(model_blob)
        data_out = self.pipeline.createXLinkOut()
        data_out.setStreamName("data_out")

        data_in.out.link(model.input)
        model.out.link(data_out.input)

    def model_load(self):
        model_blob = os.path.join(self.model_path, "model.blob")
        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()
        self.data_in = self.oak_device.getInputQueue("data_in")
        self.data_out = self.oak_device.getOutputQueue("data_out")
        with open(
            os.path.join(os.path.dirname(__file__), "postprocessing_config.json"), "r",
        ) as fp:
            postprocess_config = json.load(fp)
        self.postprocessor = PriorUtil(**postprocess_config)
        return self.pipeline

    def forward(self, data):
        results = []
        for sample in data[0]:
            nn_data = dai.NNData()
            nn_data.setLayer("input_1", sample)
            self.data_in.send(nn_data)
            assert wait_for_results(self.data_out)
            results.append(self.data_out.get())
        data[0] = results
        return data

    def add_cam_to_pipeline(self, preview_width, preview_height):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(preview_width, preview_height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setCamId(0)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        cam_queue = self.oak_device.getOutputQueue("cam_out", 1, True)
        self.data_in = self.oak_device.getInputQueue("data_in")
        self.data_out = self.oak_device.getOutputQueue("data_out")

        return cam_queue

    def to_device(self, _):
        pass

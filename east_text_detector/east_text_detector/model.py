import math
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, Point, TextPolygon

from .postprocessing import decode_predictions, non_max_suppression, rotated_rectangle


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
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.input_height, self.input_width = 320, 320

    def preprocess(self, data):
        preprocessed_data = []
        data_info = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
            height, width, _ = img.shape
            w_scale = width / float(self.input_width)
            h_scale = height / float(self.input_height)
            data_info.append((w_scale, h_scale))
            scaled_img = cv2.resize(img, (self.input_height, self.input_width))
            scaled_img = scaled_img.astype(np.float32)
            scaled_img = scaled_img.transpose((2, 0, 1))
            scaled_img = scaled_img[np.newaxis]
            preprocessed_data.append(scaled_img)
        return [preprocessed_data, data_info]

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
            scale_w, scale_h = input_info
            image_predictions = []
            for box, confidence, angle in zip(boxes, confidences, angles):
                original_w = scale_w * self.input_width
                original_h = scale_h * self.input_height
                x1 = np.clip(int(box[0] * scale_w), 0, original_w)
                y1 = np.clip(int(box[1] * scale_h), 0, original_h)
                x2 = np.clip(int(box[2] * scale_w), 0, original_w)
                y2 = np.clip(int(box[3] * scale_h), 0, original_h)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                centerX = int(x1 + width / 2)
                centerY = int(y1 + height / 2)
                rotatedRect = ((centerX, centerY), ((x2 - x1), (y2 - y1)), -angle)
                points = rotated_rectangle(rotatedRect)
                points = [Point(x=x, y=y) for x, y in points]
                image_predictions.append(TextPolygon(points=points))
            postprocessed_result.append(image_predictions)
        return postprocessed_result

    def to_device(self, device):
        pass

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
        return self.pipeline

    def forward(self, data):
        results = []
        for sample in data[0]:
            nn_data = dai.NNData()
            nn_data.setLayer("input_images", sample)
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

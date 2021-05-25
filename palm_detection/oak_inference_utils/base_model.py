import math
import os
from abc import ABC
from datetime import datetime, timedelta
from typing import Any, Tuple

import cv2
import depthai as dai
import pydantic
from modelplace_api import BaseModel, Device


def pad_img(img, pad_value, target_dims):
    h, w, _ = img.shape
    pads = [
        math.floor((target_dims[0] - h) // 2),
        math.floor((target_dims[1] - w) // 2),
    ]
    padded_img = cv2.copyMakeBorder(
        img,
        pads[0],
        int(target_dims[0] - h - pads[0]),
        pads[1],
        int(target_dims[1] - w - pads[1]),
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded_img, pads


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


class DataInfo(pydantic.BaseModel):
    scales: Tuple[float, float]
    pads: Tuple[int, int]
    original_width: int
    original_height: int


class OAKSingleStageModel(BaseModel, ABC):
    def __init__(
        self,
        model_path: str,
        input_name: str,
        input_shapes: Tuple[int, int],
        model_name: str = "",
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.input_name = input_name
        self.input_width, self.input_height = input_shapes

    def process_sample(self, image):
        data = self.preprocess([image]) if self.cam_queue is None else None
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

    def create_pipeline(self, model_blob):
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)
        data_in = self.pipeline.createXLinkIn()
        data_in.setStreamName("data_in")

        self.model_blob = self.pipeline.createNeuralNetwork()
        self.model_blob.setBlobPath(model_blob)
        data_out = self.pipeline.createXLinkOut()
        data_out.setStreamName("data_out")

        data_in.out.link(self.model_blob.input)
        self.model_blob.out.link(data_out.input)

    def model_load(self, device=Device.cpu):
        model_blob = os.path.join(self.model_path, "model.blob")
        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()
        self.data_in = self.oak_device.getInputQueue("data_in")
        self.data_out = self.oak_device.getOutputQueue("data_out")
        self.cam_queue = None

    def forward(self, data):
        results = []
        if data is not None:
            for sample in data[0]:
                nn_data = dai.NNData()
                nn_data.setLayer(self.input_name, sample)
                self.data_in.send(nn_data)
                assert wait_for_results(self.data_out)
                results.append(self.data_out.get())
            data[0] = results
        else:
            assert wait_for_results(self.data_out)
            results.append(self.data_out.get())
            data = [
                results,
                [
                    DataInfo(
                        scales=(1.0, 1.0),
                        pads=(0, 0),
                        original_width=self.input_width,
                        original_height=self.input_height,
                    ),
                ],
            ]

        return data

    def add_cam_to_pipeline(self):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(self.input_width, self.input_height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.preview.link(self.model_blob.input)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device
        del self.data_in

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.cam_queue = self.oak_device.getOutputQueue(
            "cam_out", maxSize=1, blocking=False,
        )

        self.data_out = self.oak_device.getOutputQueue("data_out")

    def get_frame_from_camera(self):
        if self.cam_queue is not None:
            return self.cam_queue.get().getData()
        else:
            raise AttributeError("Initialize camera with `add_cam_to_pipeline` method")

    def get_input_shapes(self):
        return self.input_width, self.input_height


class OAKTwoStageModel(BaseModel, ABC):
    def __init__(
        self,
        model_path: str,
        input_name: str,
        first_stage: Any,
        model_name: str = "",
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.input_name = input_name
        self.first_stage = first_stage

    def process_sample(self, image):
        data = self.preprocess([image])
        if data is None:
            return []
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

    def create_pipeline(self, model_blob):
        self.pipeline = dai.Pipeline()

        first_stage_in = self.pipeline.createXLinkIn()
        first_stage_in.setStreamName("first_stage_in")

        first_stage_nn = self.pipeline.createNeuralNetwork()
        first_stage_nn.setBlobPath(model_blob["first_stage_nn"])

        first_stage_out = self.pipeline.createXLinkOut()
        first_stage_out.setStreamName("first_stage_out")

        second_stage_in = self.pipeline.createXLinkIn()
        second_stage_in.setStreamName("second_stage_in")

        second_stage_nn = self.pipeline.createNeuralNetwork()
        second_stage_nn.setBlobPath(model_blob["second_stage_nn"])

        second_stage_out = self.pipeline.createXLinkOut()
        second_stage_out.setStreamName("second_stage_out")

        first_stage_in.out.link(first_stage_nn.input)
        first_stage_nn.out.link(first_stage_out.input)
        second_stage_in.out.link(second_stage_nn.input)
        second_stage_nn.out.link(second_stage_out.input)

    def model_load(self, device=Device.cpu):
        model_blob = {
            "first_stage_nn": os.path.join(self.model_path, "stage_1.blob"),
            "second_stage_nn": os.path.join(self.model_path, "stage_2.blob"),
        }

        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.first_stage_in = self.oak_device.getInputQueue("first_stage_in")
        self.first_stage_out = self.oak_device.getOutputQueue("first_stage_out")
        self.second_stage_in = self.oak_device.getInputQueue("second_stage_in")
        self.second_stage_out = self.oak_device.getOutputQueue("second_stage_out")
        self.cam_queue = None

    def forward(self, data):
        results = []
        for sample in data[0]:
            sample_results = []
            for face in sample:
                nn_data = dai.NNData()
                nn_data.setLayer("data", face)
                self.second_stage_in.send(nn_data)
                assert wait_for_results(self.second_stage_out)
                sample_results.append(self.second_stage_out.get())
            results.append(sample_results)
        data[0] = results
        return data

    def add_cam_to_pipeline(self):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(*self.get_input_shapes())
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.cam_queue = self.oak_device.getOutputQueue(
            "cam_out", maxSize=1, blocking=False,
        )
        self.first_stage_in = self.oak_device.getInputQueue("first_stage_in")
        self.first_stage_out = self.oak_device.getOutputQueue("first_stage_out")
        self.second_stage_in = self.oak_device.getInputQueue("second_stage_in")
        self.second_stage_out = self.oak_device.getOutputQueue("second_stage_out")

    def get_frame_from_camera(self):
        if self.cam_queue is not None:
            return self.cam_queue.get().getData()
        else:
            raise AttributeError("Initialize camera with `add_cam_to_pipeline` method")

    def get_first_stage_result(self, data):
        preprocessed_data = self.first_stage.preprocess(data)
        first_stage_output = self.first_stage.forward(
            self.first_stage_in, self.first_stage_out, preprocessed_data,
        )
        first_stage_result = self.first_stage.postprocess(first_stage_output)
        return first_stage_result

    def get_input_shapes(self):
        return self.first_stage.get_input_shapes()

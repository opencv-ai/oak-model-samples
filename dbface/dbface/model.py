import math
import os

import cv2
from modelplace_api import BaseModel, BBox, FacialLandmarks, Point, TaskType
import depthai as dai
from datetime import datetime, timedelta

from .utils import *


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=3):
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
        threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.class_names = {
            0: "face",
        }
        self.input_height, self.input_width = 480, 640

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
            height, width, _ = img.shape
            scale_y = self.input_height / height
            scale_x = self.input_width / width
            resized_image = cv2.resize(
                img,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_CUBIC,
            )
            self.resized_image = resized_image
            resized_image = resized_image.transpose((2, 0, 1))
            resized_image = resized_image[np.newaxis].astype(np.float32)
            preprocessed_data.append(resized_image)
            data_infos.append((scale_y, scale_x))
        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            scale_y, scale_x = input_info
            h, w = self.input_height, self.input_width
            original_h = int(h / scale_y)
            original_w = int(w / scale_x)
            hm = np.array(result.getLayerFp16(result.getAllLayerNames()[2])).reshape((1, 1, 120, 160))
            box = np.array(result.getLayerFp16(result.getAllLayerNames()[1])).reshape((1, 4, 120, 160))
            landmark = np.array(result.getLayerFp16(result.getAllLayerNames()[0])).reshape((1, 10, 120, 160))
            objs = detect(
                hm=hm, box=box, landmark=landmark, threshold=self.threshold, nms_iou=0.5,
            )
            image_predictions = []
            for obj in objs:
                box, confidence, landmark = obj
                image_predictions.append(
                    FacialLandmarks(
                        bbox=BBox(
                            x1=float(np.clip(box[0] / scale_x, 0, original_w)),
                            y1=float(np.clip(box[1] / scale_y, 0, original_h)),
                            x2=float(np.clip(box[2] / scale_x, 0, original_w)),
                            y2=float(np.clip(box[3] / scale_y, 0, original_h)),
                            score=float(confidence),
                            class_name=self.class_names[0],
                        ),
                        keypoints=[
                            Point(
                                x=np.clip(x / scale_x, 0, original_w),
                                y=np.clip(y / scale_y, 0, original_h),
                            )
                            for x, y in landmark
                        ],
                    ),
                )
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
            nn_data.setLayer("x", sample)
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

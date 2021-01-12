import os

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, FacialLandmarks, Point

from .face_processing import FaceProcessor, pad_img, wait_for_results


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        area_threshold: float = 0.6,
        face_bbox_pad_percent: float = 0.25,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.face_processor = FaceProcessor(threshold)
        self.area_threshold = area_threshold
        self.face_bbox_pad_percent = face_bbox_pad_percent
        self.input_height, self.input_width = 60, 60

    def preprocess(self, data):
        face_bboxes = self.get_faces(data)
        preprocessed_data = []
        preprocessed_bboxes = []
        if face_bboxes == [[]]:
            return [preprocessed_bboxes, preprocessed_data]
        for i, img in enumerate(data):
            areas = [
                (bbox.y2 - bbox.y1) * (bbox.x2 - bbox.x1) for bbox in face_bboxes[i]
            ]
            max_area = max(areas)
            preprocessed_img = []
            img_bboxes = []
            img = np.array(img)[:, :, ::-1]
            for j, face_bbox in enumerate(face_bboxes[i]):
                if areas[j] < self.area_threshold * max_area:
                    continue
                # apply padding to face bbox to increase quality of facial landmark model
                face_bbox_width = face_bbox.x2 - face_bbox.x1
                face_bbox_hight = face_bbox.y2 - face_bbox.y1
                face_bbox.x1 = max(
                    0, face_bbox.x1 - face_bbox_width * self.face_bbox_pad_percent,
                )
                face_bbox.y2 = min(
                    img.shape[0],
                    face_bbox.y2 + face_bbox_hight * self.face_bbox_pad_percent,
                )
                face_bbox.x2 = min(
                    img.shape[1],
                    face_bbox.x2 + face_bbox_width * self.face_bbox_pad_percent,
                )
                cropped_face = img[
                    int(face_bbox.y1) : int(face_bbox.y2),
                    int(face_bbox.x1) : int(face_bbox.x2),
                ]
                height, width, _ = cropped_face.shape
                if self.input_height / self.input_width < height / width:
                    scale = self.input_height / height
                else:
                    scale = self.input_width / width

                scaled_img = cv2.resize(
                    cropped_face,
                    (0, 0),
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                padded_img, pad = pad_img(
                    scaled_img, (0, 0, 0), [self.input_height, self.input_width],
                )

                padded_img = padded_img.transpose((2, 0, 1))
                padded_img = padded_img[np.newaxis].astype(np.float32)
                preprocessed_img.append(padded_img)
                img_bboxes.append(face_bbox)
            preprocessed_data.append(preprocessed_img)
            preprocessed_bboxes.append(img_bboxes)

        return [preprocessed_data, preprocessed_bboxes]

    def postprocess(self, predictions):
        postprocessed_result = []

        for results, face_bboxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for result, face_bbox in zip(results, face_bboxes):
                keypoints = np.array(result.getLayerFp16(result.getAllLayerNames()[0]))
                bbox_w, bbox_h = (
                    (face_bbox.x2 - face_bbox.x1),
                    (face_bbox.y2 - face_bbox.y1),
                )
                image_predictions.append(
                    FacialLandmarks(
                        bbox=face_bbox,
                        keypoints=[
                            Point(
                                x=face_bbox.x1 + bbox_w * x,
                                y=face_bbox.y1 + bbox_h * y,
                            )
                            for x, y in zip(keypoints[::2], keypoints[1::2])
                        ],
                    ),
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def create_pipeline(self, model_blob):
        self.pipeline = dai.Pipeline()

        face_detector_in = self.pipeline.createXLinkIn()
        face_detector_in.setStreamName("face_detector_in")

        face_detector = self.pipeline.createNeuralNetwork()
        face_detector.setBlobPath(model_blob["detector"])

        face_detector_out = self.pipeline.createXLinkOut()
        face_detector_out.setStreamName("face_detector_out")

        landmark_detector_in = self.pipeline.createXLinkIn()
        landmark_detector_in.setStreamName("landmark_detector_in")

        landmark_detector_nn = self.pipeline.createNeuralNetwork()
        landmark_detector_nn.setBlobPath(model_blob["landmark_detector"])

        landmark_detector_out = self.pipeline.createXLinkOut()
        landmark_detector_out.setStreamName("landmark_detector_out")

        face_detector_in.out.link(face_detector.input)
        face_detector.out.link(face_detector_out.input)
        landmark_detector_in.out.link(landmark_detector_nn.input)
        landmark_detector_nn.out.link(landmark_detector_out.input)

    def model_load(self):
        model_blob = {
            "detector": os.path.join(self.model_path, "stage_1.blob"),
            "landmark_detector": os.path.join(self.model_path, "stage_2.blob"),
        }

        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.face_detector_in = self.oak_device.getInputQueue("face_detector_in")
        self.face_detector_out = self.oak_device.getOutputQueue("face_detector_out")
        self.landmark_detector_in = self.oak_device.getInputQueue(
            "landmark_detector_in",
        )
        self.landmark_detector_out = self.oak_device.getOutputQueue(
            "landmark_detector_out",
        )

    def forward(self, data):
        results = []
        for sample in data[0]:
            sample_results = []
            for face in sample:
                nn_data = dai.NNData()
                nn_data.setLayer("data", face)
                self.landmark_detector_in.send(nn_data)
                assert wait_for_results(self.landmark_detector_out)
                sample_results.append(self.landmark_detector_out.get())
            results.append(sample_results)
        data[0] = results
        return data

    def process_sample(self, image):
        data = self.preprocess([image])
        if not len(data[0]):
            return []
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

    def get_faces(self, data):
        preprocessed_data = self.face_processor.preprocess(data)
        face_output = self.face_processor.forward(
            self.face_detector_in, self.face_detector_out, preprocessed_data,
        )
        face_bboxes = self.face_processor.postprocess(face_output)
        return face_bboxes

    def add_cam_to_pipeline(self, width, height):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(width, height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        cam_queue = self.oak_device.getOutputQueue("cam_out", maxSize=1, blocking=False)
        self.face_detector_in = self.oak_device.getInputQueue("face_detector_in")
        self.face_detector_out = self.oak_device.getOutputQueue("face_detector_out")
        self.landmark_detector_in = self.oak_device.getInputQueue(
            "landmark_detector_in",
        )
        self.landmark_detector_out = self.oak_device.getOutputQueue(
            "landmark_detector_out",
        )

        return cam_queue

    def to_device(self, _) -> None:
        pass

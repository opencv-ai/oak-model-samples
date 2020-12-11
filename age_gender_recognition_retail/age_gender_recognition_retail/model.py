import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from model_benchmark_api import AgeGenderLabel, BaseModel, Label, TaskType

from .face_processing import FaceProcessor, pad_img, wait_for_results


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        area_threshold: float = 0.6,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.area_threshold = area_threshold
        self.threshold = threshold
        self.classes = ["female", "male"]
        self.face_processor = FaceProcessor(threshold)

    def preprocess(self, data):
        face_bboxes = self.get_faces(data)
        preprocessed_data = []
        preprocessed_bboxes = []
        if face_bboxes == [[]]:
            return None
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
                planar_img = padded_img.flatten().astype(np.float32)
                preprocessed_img.append(planar_img)
                img_bboxes.append(face_bbox)
            preprocessed_data.append(preprocessed_img)
            preprocessed_bboxes.append(img_bboxes)

        return [preprocessed_data, preprocessed_bboxes]

    def postprocess(self, predictions):
        postprocessed_result = []

        for result, face_bboxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for face_bbox, face_result in zip(face_bboxes, result):
                age, gender_score = [
                    face_result.getLayerFp16(tensor.name)
                    for tensor in face_result.getAllLayers()
                ]
                gender_idx = np.argsort(gender_score)
                image_predictions.append(
                    AgeGenderLabel(
                        bbox=face_bbox,
                        age=int(age[0] * 100),
                        gender=[
                            Label(
                                score=gender_score[idx], class_name=self.classes[idx],
                            )
                            for idx in reversed(gender_idx)
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

        age_gender_in = self.pipeline.createXLinkIn()
        age_gender_in.setStreamName("age_gender_in")

        age_gender_nn = self.pipeline.createNeuralNetwork()
        age_gender_nn.setBlobPath(model_blob["age_gender"])

        age_gender_out = self.pipeline.createXLinkOut()
        age_gender_out.setStreamName("age_gender_out")

        face_detector_in.out.link(face_detector.input)
        face_detector.out.link(face_detector_out.input)
        age_gender_in.out.link(age_gender_nn.input)
        age_gender_nn.out.link(age_gender_out.input)

    def model_load(self):
        self.task_type = TaskType.age_gender_recognition

        model_blob = {
            "detector": os.path.join(self.model_path, "stage_1.blob"),
            "age_gender": os.path.join(self.model_path, "stage_2.blob"),
        }

        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.face_detector_in = self.oak_device.getInputQueue("face_detector_in")
        self.face_detector_out = self.oak_device.getOutputQueue("face_detector_out")
        self.age_gender_in = self.oak_device.getInputQueue("age_gender_in")
        self.age_gender_out = self.oak_device.getOutputQueue("age_gender_out")

        self.input_width, self.input_height = (
            62,
            62,
        )

    def forward(self, data, stage="age-gender"):
        results = []
        for sample in data[0]:
            sample_results = []
            for face in sample:
                nn_data = dai.NNData()
                nn_data.setLayer("data", face)
                self.age_gender_in.send(nn_data)
                assert wait_for_results(self.age_gender_out)
                sample_results.append(self.age_gender_out.get())
            results.append(sample_results)
        data[0] = results
        return data

    def to_device(self, device):
        pass

    def process_sample(self, image):
        data = self.preprocess([image])
        if data is None:
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
        cam.setCamId(0)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        cam_queue = self.oak_device.getOutputQueue("cam_out", 1, True)
        self.face_detector_in = self.oak_device.getInputQueue("face_detector_in")
        self.face_detector_out = self.oak_device.getOutputQueue("face_detector_out")
        self.age_gender_in = self.oak_device.getInputQueue("age_gender_in")
        self.age_gender_out = self.oak_device.getOutputQueue("age_gender_out")

        return cam_queue

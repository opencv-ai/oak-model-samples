import os

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, EmotionLabel, Label

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
        self.face_processor = FaceProcessor(threshold)
        self.area_threshold = area_threshold
        self.classes = ["neutral", "happy", "sad", "surprise", "anger"]
        self.input_height, self.input_width = 64, 64

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
                if face_bbox.x2 - face_bbox.x1 < self.input_width:
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
                padded_img = padded_img[np.newaxis].astype(np.float32)
                preprocessed_img.append(padded_img)
                img_bboxes.append(face_bbox)
            preprocessed_data.append(preprocessed_img)
            preprocessed_bboxes.append(img_bboxes)

        return [preprocessed_data, preprocessed_bboxes]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, face_bboxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for face_bbox, result_emotion_prob in zip(face_bboxes, result):
                emotions_probs_list = result_emotion_prob.getLayerFp16(
                    result_emotion_prob.getAllLayerNames()[0],
                )
                emotions_index = np.argsort(emotions_probs_list)
                image_predictions.append(
                    EmotionLabel(
                        bbox=face_bbox,
                        emotion=[
                            Label(
                                score=emotions_probs_list[idx],
                                class_name=self.classes[idx],
                            )
                            for idx in reversed(emotions_index)
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

        emotion_recognition_in = self.pipeline.createXLinkIn()
        emotion_recognition_in.setStreamName("emotion_recognition_in")

        emotion_recognition_nn = self.pipeline.createNeuralNetwork()
        emotion_recognition_nn.setBlobPath(model_blob["emotion_recognition"])

        emotion_recognition_out = self.pipeline.createXLinkOut()
        emotion_recognition_out.setStreamName("emotion_recognition_out")

        face_detector_in.out.link(face_detector.input)
        face_detector.out.link(face_detector_out.input)
        emotion_recognition_in.out.link(emotion_recognition_nn.input)
        emotion_recognition_nn.out.link(emotion_recognition_out.input)

    def model_load(self):

        model_blob = {
            "detector": os.path.join(self.model_path, "stage_1.blob"),
            "emotion_recognition": os.path.join(self.model_path, "stage_2.blob"),
        }

        self.create_pipeline(model_blob)

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        self.face_detector_in = self.oak_device.getInputQueue("face_detector_in")
        self.face_detector_out = self.oak_device.getOutputQueue("face_detector_out")
        self.emotion_recognition_in = self.oak_device.getInputQueue(
            "emotion_recognition_in",
        )
        self.emotion_recognition_out = self.oak_device.getOutputQueue(
            "emotion_recognition_out",
        )

    def forward(self, data):
        results = []
        for sample in data[0]:
            sample_results = []
            for face in sample:
                nn_data = dai.NNData()
                nn_data.setLayer("data", face)
                self.emotion_recognition_in.send(nn_data)
                assert wait_for_results(self.emotion_recognition_out)
                sample_results.append(self.emotion_recognition_out.get())
            results.append(sample_results)
        data[0] = results
        return data

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
        self.emotion_recognition_in = self.oak_device.getInputQueue(
            "emotion_recognition_in",
        )
        self.emotion_recognition_out = self.oak_device.getOutputQueue(
            "emotion_recognition_out",
        )

        return cam_queue

    def to_device(self, _) -> None:
        pass

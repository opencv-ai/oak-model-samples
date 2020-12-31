import math
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, BBox, FacialLandmarks, Point, TaskType


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


class BlazeDecoder:
    def __init__(self, score_threshold=0.75, iou_threshold=0.3):
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = score_threshold
        self.min_suppression_threshold = iou_threshold

    def tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.
        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.
        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert len(raw_box_tensor.shape) == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert len(raw_box_tensor.shape) == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clip(-thresh, thresh)
        detection_scores = 1 / (1 + np.exp(-raw_score_tensor)).squeeze(axis=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
            output_detections.append(np.concatenate((boxes, scores), axis=-1))

        # return output_detections
        return [self.nms(output_detection) for output_detection in output_detections]

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = np.zeros(raw_boxes.shape)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.0  # ymin
        boxes[..., 1] = x_center - w / 2.0  # xmin
        boxes[..., 2] = y_center + h / 2.0  # ymax
        boxes[..., 3] = x_center + w / 2.0  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = (
                raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            )
            keypoint_y = (
                raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3]
                + anchors[:, 1]
            )
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def nms(self, detections):
        bboxes = np.concatenate(
            [detections[:, 0:2], detections[:, 2:4] - detections[:, 0:2]], axis=1,
        )
        indeces = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            detections[:, 16].tolist(),
            self.min_score_thresh,
            self.min_suppression_threshold,
        )
        return [detections[id[0]] for id in indeces]


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


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.4,
        iou_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.class_names = {
            0: "background",
            1: "person",
        }
        self.decoder = BlazeDecoder(threshold, iou_threshold)
        self.input_height, self.input_width = 128, 128

    def nms(self, detections):
        bboxes = [[box.xmin, box.ymin, box.w, box.h] for box in detections]
        scores = [box.confidence for box in detections]
        indeces = cv2.dnn.NMSBoxes(bboxes, scores, self.threshold, self.iou_threshold)
        return [detections[id[0]] for id in indeces]

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
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
            padded_img = padded_img / 127.5 - 1
            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float16)
            preprocessed_data.append(padded_img)
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pads = input_info
            h, w = self.input_height, self.input_width
            raw_scores = np.concatenate(
                [
                    np.array(
                        result.getLayerFp16(
                            "StatefulPartitionedCall/functional_1/tf_op_layer_classificators_1/classificators_1",
                        ),
                    ).reshape((-1, 512, 1)),
                    np.array(
                        result.getLayerFp16(
                            "StatefulPartitionedCall/functional_1/tf_op_layer_classificators_2/classificators_2",
                        ),
                    ).reshape((-1, 384, 1)),
                ],
                axis=1,
            )
            raw_bboxes = np.concatenate(
                [
                    np.array(
                        result.getLayerFp16(
                            "StatefulPartitionedCall/functional_1/tf_op_layer_regressors_1/regressors_1",
                        ),
                    ).reshape((-1, 512, 16)),
                    np.array(
                        result.getLayerFp16(
                            "StatefulPartitionedCall/functional_1/tf_op_layer_regressors_2/regressors_2",
                        ),
                    ).reshape((-1, 384, 16)),
                ],
                axis=1,
            )
            detections = self.decoder.tensors_to_detections(
                raw_bboxes, raw_scores, self.anchors,
            )[0]
            image_predictions = []
            for detection in detections:
                if detection[16] > self.threshold:
                    image_predictions.append(
                        FacialLandmarks(
                            bbox=BBox(
                                x1=(float(detection[1]) * w - pads[1]) / scale,
                                y1=(float(detection[0]) * h - pads[0]) / scale,
                                x2=(float(detection[3]) * w - pads[1]) / scale,
                                y2=(float(detection[2]) * h - pads[0]) / scale,
                                score=float(detection[16]),
                                class_name="Face",
                            ),
                            keypoints=[
                                Point(
                                    x=int(
                                        (detection[4 + k * 2] * w - pads[1]) / scale,
                                    ),
                                    y=int(
                                        (detection[4 + k * 2 + 1] * h - pads[0])
                                        / scale,
                                    ),
                                )
                                for k in range(6)
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
        self.anchors = np.load(os.path.join(self.model_path, "anchors.npy"))

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
            nn_data.setLayer("input", sample)
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

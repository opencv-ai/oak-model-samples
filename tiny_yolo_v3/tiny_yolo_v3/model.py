import math
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, BBox


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


class DetectionObject:
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0
    w = 0
    h = 0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.w = int(w * w_scale)
        self.h = int(h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        iou_threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.class_names = {
            0: "background",
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorbike",
            5: "aeroplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light",
            11: "fire hydrant",
            12: "stop sign",
            13: "parking meter",
            14: "bench",
            15: "bird",
            16: "cat",
            17: "dog",
            18: "horse",
            19: "sheep",
            20: "cow",
            21: "elephant",
            22: "bear",
            23: "zebra",
            24: "giraffe",
            25: "backpack",
            26: "umbrella",
            27: "handbag",
            28: "tie",
            29: "suitcase",
            30: "frisbee",
            31: "skis",
            32: "snowboard",
            33: "sports ball",
            34: "kite",
            35: "baseball bat",
            36: "baseball glove",
            37: "skateboard",
            38: "surfboard",
            39: "tennis racket",
            40: "bottle",
            41: "wine glass",
            42: "cup",
            43: "fork",
            44: "knife",
            45: "spoon",
            46: "bowl",
            47: "banana",
            48: "apple",
            49: "sandwich",
            50: "orange",
            51: "broccoli",
            52: "carrot",
            53: "hot dog",
            54: "pizza",
            55: "donut",
            56: "cake",
            57: "chair",
            58: "sofa",
            59: "pottedplant",
            60: "bed",
            61: "diningtable",
            62: "toilet",
            63: "tvmonitor",
            64: "laptop",
            65: "mouse",
            66: "remote",
            67: "keyboard",
            68: "cell phone",
            69: "microwave",
            70: "oven",
            71: "toaster",
            72: "sink",
            73: "refrigerator",
            74: "book",
            75: "clock",
            76: "vase",
            77: "scissors",
            78: "teddy bear",
            79: "hair drier",
            80: "toothbrush",
        }
        self.anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
        self.yolo_scale_13 = 13
        self.yolo_scale_26 = 26
        self.input_height, self.input_width = 416, 416

    @staticmethod
    def entryindex(side, lcoords, lclasses, location, entry):
        n = int(location / (side * side))
        loc = location % (side * side)
        return int(
            n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc,
        )

    def parse_yolov3_output(
        self, blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold,
    ):
        num = 3
        coords = 4
        objects = []
        out_blob_h = blob.shape[2]
        side = out_blob_h
        anchor_offset = 0
        if len(self.anchors) == 12:
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 0
        else:
            assert "Wrong anchors amount for Tiny Yolov3"
        side_square = side * side
        output_blob = blob.flatten()

        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(num):
                obj_index = self.entryindex(
                    side,
                    coords,
                    len(self.class_names) - 1,
                    n * side * side + i,
                    coords,
                )
                box_index = self.entryindex(
                    side, coords, len(self.class_names) - 1, n * side * side + i, 0,
                )
                scale = output_blob[obj_index]
                if scale < threshold:
                    continue
                x = (
                    (col + output_blob[box_index + 0 * side_square])
                    / side
                    * resized_im_w
                )
                y = (
                    (row + output_blob[box_index + 1 * side_square])
                    / side
                    * resized_im_h
                )
                height = (
                    math.exp(output_blob[box_index + 3 * side_square])
                    * self.anchors[anchor_offset + 2 * n + 1]
                )
                width = (
                    math.exp(output_blob[box_index + 2 * side_square])
                    * self.anchors[anchor_offset + 2 * n]
                )
                for j in range(len(self.class_names) - 1):
                    class_index = self.entryindex(
                        side,
                        coords,
                        len(self.class_names) - 1,
                        n * side_square + i,
                        coords + 1 + j,
                    )
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    obj = DetectionObject(
                        x,
                        y,
                        height,
                        width,
                        j,
                        prob,
                        (original_im_h / resized_im_h),
                        (original_im_w / resized_im_w),
                    )
                    objects.append(obj)
        return objects

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
            height, width, _ = img.shape
            resized_image = cv2.resize(
                img,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_CUBIC,
            )
            resized_image = resized_image.transpose((2, 0, 1))
            resized_image = resized_image[np.newaxis].astype(np.float32)
            preprocessed_data.append(resized_image)
            data_infos.append((height, width))

        return [preprocessed_data, data_infos]

    def nms(self, detections):
        bboxes = [[box.xmin, box.ymin, box.w, box.h] for box in detections]
        scores = [box.confidence for box in detections]
        indeces = cv2.dnn.NMSBoxes(bboxes, scores, self.threshold, self.iou_threshold)
        return [detections[id[0]] for id in indeces]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            objects = []
            original_h, original_w = input_info
            h, w = self.input_height, self.input_width
            output_shapes = [(-1, 255, 26, 26), (-1, 255, 13, 13)]
            for output_name, output_shape in zip(
                result.getAllLayerNames(), output_shapes,
            ):
                objects.extend(
                    self.parse_yolov3_output(
                        np.array(result.getLayerFp16(output_name)).reshape(
                            output_shape,
                        ),
                        h,
                        w,
                        original_h,
                        original_w,
                        self.iou_threshold,
                    ),
                )
            objects = self.nms(objects)
            image_predictions = []
            for obj in objects:
                image_predictions.append(
                    BBox(
                        x1=float(np.clip(obj.xmin, 0, original_w)),
                        y1=float(np.clip(obj.ymin, 0, original_h)),
                        x2=float(np.clip(obj.xmax, 0, original_w)),
                        y2=float(np.clip(obj.ymax, 0, original_h)),
                        score=float(obj.confidence),
                        class_name=self.class_names[int(obj.class_id) + 1],
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
            nn_data.setLayer("inputs", sample)
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
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = self.pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        del self.oak_device

        self.oak_device = dai.Device(self.pipeline)
        self.oak_device.startPipeline()

        cam_queue = self.oak_device.getOutputQueue("cam_out", maxSize=1, blocking=False)
        self.data_in = self.oak_device.getInputQueue("data_in")
        self.data_out = self.oak_device.getOutputQueue("data_out")

        return cam_queue

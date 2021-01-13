import math
import os
from datetime import datetime, timedelta
from math import exp as exp

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

    def __init__(self, xmin, ymin, xmax, ymax, class_id, confidence):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_id = class_id
        self.confidence = confidence


yolo_params = {
    "detector/yolo-v4-tiny/Conv_17/BiasAdd/YoloRegion": {
        "anchors": "10,14,23,27,37,58,81,82,135,169,344,319",
        "axis": "1",
        "classes": "80",
        "coords": "4",
        "do_softmax": "0",
        "end_axis": "3",
        "mask": "3,4,5",
        "num": "6",
        "originalLayersNames": "detector/yolo-v4-tiny/Conv_17/BiasAdd/YoloRegion",
    },
    "detector/yolo-v4-tiny/Conv_20/BiasAdd/YoloRegion": {
        "anchors": "10,14,23,27,37,58,81,82,135,169,344,319",
        "axis": "1",
        "classes": "80",
        "coords": "4",
        "do_softmax": "0",
        "end_axis": "3",
        "mask": "1,2,3",
        "num": "6",
        "originalLayersNames": "detector/yolo-v4-tiny/Conv_20/BiasAdd/YoloRegion",
    },
}


class YoloParams:
    def __init__(self, param, side):
        self.num = 3 if "num" not in param else int(param["num"])
        self.coords = 4 if "coords" not in param else int(param["coords"])
        self.classes = 80 if "classes" not in param else int(param["classes"])
        self.side = side
        self.anchors = (
            [
                10.0,
                13.0,
                16.0,
                30.0,
                33.0,
                23.0,
                30.0,
                61.0,
                62.0,
                45.0,
                59.0,
                119.0,
                116.0,
                90.0,
                156.0,
                198.0,
                373.0,
                326.0,
            ]
            if "anchors" not in param
            else [float(a) for a in param["anchors"].split(",")]
        )
        self.isYoloV3 = False

        if param.get("mask"):
            mask = [int(idx) for idx in param["mask"].split(",")]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True


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
        self.input_height, self.input_width = 416, 416

    @staticmethod
    def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w):
        xmin = int((x - width / 2) * im_w)
        ymin = int((y - height / 2) * im_h)
        xmax = int(xmin + width * im_w)
        ymax = int(ymin + height * im_h)
        return DetectionObject(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            class_id=class_id,
            confidence=confidence,
        )

    def parse_yolo_region(
        self, predictions, resized_image_shape, original_im_shape, params, threshold,
    ):
        _, _, out_blob_h, out_blob_w = predictions.shape
        assert out_blob_w == out_blob_h, (
            "Invalid size of output blob. It sould be in NCHW layout and height should "
            "be equal to width. Current height = {}, current width = {}"
            "".format(out_blob_h, out_blob_w)
        )
        orig_im_h, orig_im_w = original_im_shape
        objects = list()
        resized_image_h, resized_image_w = resized_image_shape
        size_normalizer = (
            (resized_image_w, resized_image_h)
            if params.isYoloV3
            else (params.side, params.side)
        )
        bbox_size = params.coords + 1 + params.classes
        for row, col, n in np.ndindex(params.side, params.side, params.num):
            bbox = predictions[0, n * bbox_size : (n + 1) * bbox_size, row, col]
            x, y, width, height, object_probability = bbox[:5]
            class_probabilities = bbox[5:]
            if object_probability < threshold:
                continue
            x = (col + x) / params.side
            y = (row + y) / params.side
            try:
                width = exp(width)
                height = exp(height)
            except OverflowError:
                continue
            width = width * params.anchors[2 * n] / size_normalizer[0]
            height = height * params.anchors[2 * n + 1] / size_normalizer[1]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id] * object_probability
            if confidence < threshold:
                continue
            objects.append(
                self.scale_bbox(
                    x=x,
                    y=y,
                    height=height,
                    width=width,
                    class_id=class_id,
                    confidence=confidence,
                    im_h=orig_im_h,
                    im_w=orig_im_w,
                ),
            )
        return objects

    def get_objects(
        self,
        output,
        net,
        new_frame_height_width,
        source_height_width,
        prob_threshold,
        is_proportional,
    ):
        objects = list()

        for layer_name, out_blob in output.items():
            out_blob = out_blob.buffer.reshape(
                net.layers[net.layers[layer_name].parents[0]].out_data[0].shape,
            )
            layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
            objects += self.parse_yolo_region(
                out_blob,
                new_frame_height_width,
                source_height_width,
                layer_params,
                prob_threshold,
                is_proportional,
            )

        return objects

    def filter_objects(self, objects, iou_threshold, prob_threshold):
        objects = sorted(objects, key=lambda obj: obj["confidence"], reverse=True)
        for i in range(len(objects)):
            if objects[i]["confidence"] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if self.iou(objects[i], objects[j]) > iou_threshold:
                    objects[j]["confidence"] = 0

        return tuple(obj for obj in objects if obj["confidence"] >= prob_threshold)

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
            height, width, _ = img.shape
            resized_image = cv2.resize(
                img,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            resized_image = resized_image.transpose((2, 0, 1))
            resized_image = resized_image[np.newaxis].astype(np.float32)
            preprocessed_data.append(resized_image)
            data_infos.append((height, width))

        return [preprocessed_data, data_infos]

    @staticmethod
    def iou(box_1, box_2):
        width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(
            box_1.xmin, box_2.xmin,
        )
        height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(
            box_1.ymin, box_2.ymin,
        )
        if width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0:
            area_of_overlap = 0.0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
        box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union <= 0.0:
            retval = 0.0
        else:
            retval = area_of_overlap / area_of_union
        return retval

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            objects = []
            original_h, original_w = input_info
            output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26)]
            for output_name, output_shape in zip(
                result.getAllLayerNames(), output_shapes,
            ):
                layer_params = YoloParams(yolo_params[output_name], output_shape[2])
                output = np.array(result.getLayerFp16(output_name)).reshape(
                    output_shape,
                )
                objects += self.parse_yolo_region(
                    output,
                    (self.input_height, self.input_width),
                    (original_h, original_w),
                    layer_params,
                    self.threshold,
                )
            boxes = []
            confidences = []
            class_ids = []
            image_predictions = []
            for i in range(len(objects)):
                if objects[i].confidence == 0.0:
                    continue
                for j in range(i + 1, len(objects)):
                    if self.iou(objects[i], objects[j]) >= self.iou_threshold:
                        if objects[i].confidence < objects[j].confidence:
                            objects[i], objects[j] = objects[j], objects[i]
                        objects[j].confidence = 0.0

            for obj in objects:
                boxes.append([obj.xmin, obj.ymin, obj.xmax, obj.ymax])
                confidences.append(obj.confidence)
                class_ids.append(int(obj.class_id) + 1)
            boxes = np.array(boxes, dtype=np.float)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf > self.threshold:
                    image_predictions.append(
                        BBox(
                            x1=float(np.clip(box[0], 0, original_w)),
                            y1=float(np.clip(box[1], 0, original_h)),
                            x2=float(np.clip(box[2], 0, original_w)),
                            y2=float(np.clip(box[3], 0, original_h)),
                            score=float(conf),
                            class_name=self.class_names[class_id],
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

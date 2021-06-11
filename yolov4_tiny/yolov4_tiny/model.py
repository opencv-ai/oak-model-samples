import json
import os
from math import exp

import cv2
import numpy as np
from modelplace_api import BBox

from oak_inference_utils import DataInfo, OAKSingleStageModel


class DetectionObject:
    def __init__(self, x, y, h, w, class_id, confidence, im_h, im_w):
        self.xmin = int((x - w / 2) * im_w)
        self.ymin = int((y - h / 2) * im_h)
        self.xmax = int(self.xmin + w * im_w)
        self.ymax = int(self.ymin + h * im_h)
        self.w = int(w * im_w)
        self.h = int(h * im_h)
        self.class_id = class_id
        self.confidence = confidence


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

            masked_anchors = []
            for idx in mask:
                masked_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = masked_anchors

            self.isYoloV3 = True


class InferenceModel(OAKSingleStageModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        iou_threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="inputs",
            input_shapes=(416, 416),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
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
        params_file = os.path.join(os.path.dirname(__file__), "yolov4_params.json")
        with open(params_file) as json_file:
            self.yolo_params = json.load(json_file)

    @staticmethod
    def parse_yolo_region(
        predictions, resized_image_shape, original_im_shape, params, threshold,
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
                DetectionObject(
                    x, y, height, width, class_id, confidence, orig_im_h, orig_im_w,
                ),
            )
        return objects

    def nms(self, detections):
        bboxes = [[box.xmin, box.ymin, box.w, box.h] for box in detections]
        scores = [float(box.confidence) for box in detections]
        indeces = cv2.dnn.NMSBoxes(bboxes, scores, self.threshold, self.iou_threshold)
        return [detections[id[0]] for id in indeces]

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
            data_infos.append(
                DataInfo(
                    scales=(0, 0),
                    pads=(0, 0),
                    original_width=width,
                    original_height=height,
                ),
            )

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
            original_h, original_w = (
                input_info.original_height,
                input_info.original_width,
            )
            output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26)]
            for output_name, output_shape in zip(
                result.getAllLayerNames(), output_shapes,
            ):
                layer_params = YoloParams(
                    self.yolo_params[output_name], output_shape[2],
                )
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
            objects = self.nms(objects)
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
                            x1=int(np.clip(box[0], 0, original_w)),
                            y1=int(np.clip(box[1], 0, original_h)),
                            x2=int(np.clip(box[2], 0, original_w)),
                            y2=int(np.clip(box[3], 0, original_h)),
                            score=float(conf),
                            class_name=self.class_names[class_id],
                        ),
                    )
            postprocessed_result.append(image_predictions)
        return postprocessed_result

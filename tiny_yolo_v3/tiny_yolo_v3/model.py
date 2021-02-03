import math

import cv2
import numpy as np
from modelplace_api import BBox

from oak_inference_utils import DataInfo, OAKSingleStageModel


class DetectionObject:
    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.w = int(w * w_scale)
        self.h = int(h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


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
            data_infos.append(
                DataInfo(
                    scales=(0, 0),
                    pads=(0, 0),
                    original_width=width,
                    original_height=height,
                ),
            )

        return [preprocessed_data, data_infos]

    def nms(self, detections):
        bboxes = [[box.xmin, box.ymin, box.w, box.h] for box in detections]
        scores = [box.confidence for box in detections]
        indices = cv2.dnn.NMSBoxes(bboxes, scores, self.threshold, self.iou_threshold)
        return [detections[id[0]] for id in indices]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            objects = []
            original_h, original_w = (
                input_info.original_height,
                input_info.original_width,
            )
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
                        x1=int(np.clip(obj.xmin, 0, original_w)),
                        y1=int(np.clip(obj.ymin, 0, original_h)),
                        x2=int(np.clip(obj.xmax, 0, original_w)),
                        y2=int(np.clip(obj.ymax, 0, original_h)),
                        score=float(obj.confidence),
                        class_name=self.class_names[int(obj.class_id) + 1],
                    ),
                )
            postprocessed_result.append(image_predictions)
        return postprocessed_result

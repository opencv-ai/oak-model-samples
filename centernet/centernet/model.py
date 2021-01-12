# flake8: noqa
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, BBox
from numpy.lib.stride_tricks import as_strided


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=3):
            return False
    return True


class InferenceModel(BaseModel):
    orig_coco_label_map = {
        0: "background",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
    list_classes = list(orig_coco_label_map.values())[1:]
    class_names = dict(zip(list(range(len(list_classes))), list_classes))

    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.input_width, self.input_height = 512, 512
        self._threshold = threshold

    @staticmethod
    def get_affine_transform(center, scale, rot, output_size, inv=False):
        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs
            return src_result

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w, dst_h = output_size

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        dst = np.zeros((3, 2), dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :], src[1, :] = center, center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    @staticmethod
    def _gather_feat(feat, ind):
        dim = feat.shape[1]
        ind = np.expand_dims(ind, axis=1)
        ind = np.repeat(ind, dim, axis=1)
        feat = feat[ind, np.arange(feat.shape[1])]
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = np.transpose(feat, (1, 2, 0))
        feat = feat.reshape((-1, feat.shape[2]))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, K=40):
        cat, _, width = scores.shape

        scores = scores.reshape((cat, -1))
        topk_inds = np.argpartition(scores, -K, axis=1)[:, -K:]
        topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

        topk_ys = (topk_inds / width).astype(np.int32).astype(np.float)
        topk_xs = (topk_inds % width).astype(np.int32).astype(np.float)

        topk_scores = topk_scores.reshape((-1))
        topk_ind = np.argpartition(topk_scores, -K)[-K:]
        topk_score = topk_scores[topk_ind]
        topk_clses = topk_ind / K
        topk_inds = self._gather_feat(topk_inds.reshape((-1, 1)), topk_ind).reshape(
            (K),
        )
        topk_ys = self._gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((K))
        topk_xs = self._gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((K))

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _nms(self, heat, kernel=3):
        def max_pool2d(A, kernel_size, padding=1, stride=1):
            A = np.pad(A, padding, mode="constant")
            output_shape = (
                (A.shape[0] - kernel_size) // stride + 1,
                (A.shape[1] - kernel_size) // stride + 1,
            )
            kernel_size = (kernel_size, kernel_size)
            A_w = as_strided(
                A,
                shape=output_shape + kernel_size,
                strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
            )
            A_w = A_w.reshape(-1, *kernel_size)

            return A_w.max(axis=(1, 2)).reshape(output_shape)

        pad = (kernel - 1) // 2

        hmax = np.array([max_pool2d(channel, kernel, pad) for channel in heat])
        keep = hmax == heat
        return heat * keep

    def _transform_preds(self, coords, center, scale, output_size):
        def affine_transform(pt, t):
            new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
            new_pt = np.dot(t, new_pt)
            return new_pt[:2]

        target_coords = np.zeros(coords.shape)
        trans = self.get_affine_transform(center, scale, 0, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    def _transform(self, dets, center, scale, height, width):
        dets[:, :2] = self._transform_preds(
            dets[:, 0:2], center, scale, (width, height),
        )
        dets[:, 2:4] = self._transform_preds(
            dets[:, 2:4], center, scale, (width, height),
        )
        return dets

    def preprocess(self, data):
        preprocessed_data = []
        image_sizes = []
        for image in data:
            image = np.array(image)
            height, width = image.shape[0:2]
            center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            scale = max(height, width)

            trans_input = self.get_affine_transform(
                center, scale, 0, [self.input_width, self.input_height],
            )
            resized_image = cv2.resize(image, (width, height))
            inp_image = cv2.warpAffine(
                resized_image,
                trans_input,
                (self.input_width, self.input_height),
                flags=cv2.INTER_LINEAR,
            )
            inp_image = inp_image[:, :, ::-1]
            inp_image = np.transpose(inp_image, (2, 0, 1))
            inp_image = inp_image[np.newaxis].astype(np.uint8)
            image_sizes.append([height, width])
            preprocessed_data.append(inp_image)

        return preprocessed_data, image_sizes

    def postprocess(self, predictions):
        postprocessed_detections = []
        for prediction, input_shape in zip(predictions[0], predictions[1]):
            image_predictions = []
            heat, reg, wh = prediction
            heat = np.exp(heat) / (1 + np.exp(heat))
            height, width = heat.shape[1:3]
            num_predictions = 100

            heat = self._nms(heat)
            scores, inds, clses, ys, xs = self._topk(heat, K=num_predictions)
            reg = self._tranpose_and_gather_feat(reg, inds)

            reg = reg.reshape((num_predictions, 2))
            xs = xs.reshape((num_predictions, 1)) + reg[:, 0:1]
            ys = ys.reshape((num_predictions, 1)) + reg[:, 1:2]

            wh = self._tranpose_and_gather_feat(wh, inds)
            wh = wh.reshape((num_predictions, 2))
            clses = clses.reshape((num_predictions, 1))
            scores = scores.reshape((num_predictions, 1))
            bboxes = np.concatenate(
                (
                    xs - wh[..., 0:1] / 2,
                    ys - wh[..., 1:2] / 2,
                    xs + wh[..., 0:1] / 2,
                    ys + wh[..., 1:2] / 2,
                ),
                axis=1,
            )
            detections = np.concatenate((bboxes, scores, clses), axis=1)
            mask = detections[..., 4] >= self._threshold
            filtered_detections = detections[mask]
            scale = max(input_shape)
            center = np.array(input_shape[:2]) / 2.0
            boxes = self._transform(
                filtered_detections, np.flip(center, 0), scale, height, width,
            )
            for box in boxes:
                image_predictions.append(
                    BBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        score=float(box[4]),
                        class_name=self.class_names[int(box[5])],
                    ),
                )
            postprocessed_detections.append(image_predictions)

        return postprocessed_detections

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
            nn_data.setLayer("input.1", sample)
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

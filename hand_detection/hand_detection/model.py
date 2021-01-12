import math
import os
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, Joint, Link, Pose


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


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
    kpt_names = [
        "Wrist",
        "TMCP",
        "IMCP",
        "MMCP",
        "RMCP",
        "PMCP",
        "TPIP",
        "TDIP",
        "TTIP",
        "IPIP",
        "IDIP",
        "ITIP",
        "MPIP",
        "MDIP",
        "MTIP",
        "RPIP",
        "RDIP",
        "RTIP",
        "PPIP",
        "PDIP",
        "PTIP",
    ]
    model_part_idx = {b: a for a, b in enumerate(kpt_names)}
    coco_part_labels = [
        "Wrist",
        "TMCP",
        "IMCP",
        "MMCP",
        "RMCP",
        "PMCP",
        "TPIP",
        "TDIP",
        "TTIP",
        "IPIP",
        "IDIP",
        "ITIP",
        "MPIP",
        "MDIP",
        "MTIP",
        "RPIP",
        "RDIP",
        "RTIP",
        "PPIP",
        "PDIP",
        "PTIP",
    ]
    coco_part_idx = {b: a for a, b in enumerate(coco_part_labels)}
    coco_part_orders = [
        ("Wrist", "TMCP"),
        ("TMCP", "IMCP"),
        ("IMCP", "MMCP"),
        ("MMCP", "RMCP"),
        ("Wrist", "PMCP"),
        ("PMCP", "TPIP"),
        ("TPIP", "TDIP"),
        ("TDIP", "TTIP"),
        ("Wrist", "IPIP"),
        ("IPIP", "IDIP"),
        ("IDIP", "ITIP"),
        ("ITIP", "MPIP"),
        ("Wrist", "MDIP"),
        ("MDIP", "MTIP"),
        ("MTIP", "RPIP"),
        ("RPIP", "RDIP"),
        ("Wrist", "RTIP"),
        ("RTIP", "PPIP"),
        ("PPIP", "PDIP"),
        ("PDIP", "PTIP"),
    ]

    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.input_height, self.input_width = 256, 256

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
            scaled_img = scaled_img / 128 - 1
            padded_img, pad = pad_img(
                scaled_img, (0, 0, 0), [self.input_height, self.input_width],
            )
            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float16)
            preprocessed_data.append(padded_img)
            data_infos.append((scale, pad))
        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pad = input_info
            # points = np.array(result.getLayerFp16(
            #     'StatefulPartitionedCall/functional_1/tf_op_layer_ld_21_3d/ld_21_3d')).reshape(
            #     -1, 3,
            # )

            points = np.array(result.getLayerFp16("ld_21_2d")).reshape(-1, 2)
            points[:, 0] = (points[:, 0] - pad[1]) / scale
            points[:, 1] = (points[:, 1] - pad[0]) / scale
            postprocessed_result.append(
                [
                    Pose(
                        score=1.0,
                        links=self.create_links(points, self.model_part_idx),
                        skeleton_parts=self.coco_part_labels,
                    ),
                ],
            )
        return postprocessed_result

    def create_links(self, skeleton, class_map):
        links = []
        for pair in self.coco_part_orders:
            joint_a_class_name = pair[0]
            joint_b_class_name = pair[1]
            link = Link(
                joint_a=Joint(
                    x=int(skeleton[class_map[joint_a_class_name]][0]),
                    y=int(skeleton[class_map[joint_a_class_name]][1]),
                    score=float(1),
                    class_name=str(joint_a_class_name),
                ),
                joint_b=Joint(
                    x=int(skeleton[class_map[joint_b_class_name]][0]),
                    y=int(skeleton[class_map[joint_b_class_name]][1]),
                    score=float(1),
                    class_name=str(joint_b_class_name),
                ),
            )
            links.append(link)
        return links

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
            nn_data.setLayer("data", sample)
            self.data_in.send(nn_data)
            assert wait_for_results(self.data_out)
            results.append(self.data_out.get())
        data[0] = results
        return data

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

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

    def to_device(self, _) -> None:
        pass

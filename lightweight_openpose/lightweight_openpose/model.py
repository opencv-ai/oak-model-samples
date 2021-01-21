import math
import os
from datetime import datetime, timedelta
from operator import itemgetter

import cv2
import depthai as dai
import numpy as np
from modelplace_api import BaseModel, Joint, Link, Pose

from .utils import getKeypoints, getPersonwiseKeypoints, getValidPairs


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
        "nose",
        "sho_r",
        "elb_r",
        "wri_r",
        "sho_l",
        "elb_l",
        "wri_l",
        "hip_r",
        "kne_r",
        "ank_r",
        "hip_l",
        "kne_l",
        "ank_l",
        "eye_r",
        "eye_l",
        "ear_r",
        "ear_l",
    ]
    model_part_idx = {b: a for a, b in enumerate(kpt_names)}
    coco_part_labels = [
        "nose",
        "eye_l",
        "eye_r",
        "ear_l",
        "ear_r",
        "sho_l",
        "sho_r",
        "elb_l",
        "elb_r",
        "wri_l",
        "wri_r",
        "hip_l",
        "hip_r",
        "kne_l",
        "kne_r",
        "ank_l",
        "ank_r",
    ]
    coco_part_orders = [
        ("nose", "eye_l"),
        ("eye_l", "eye_r"),
        ("eye_r", "nose"),
        ("eye_l", "ear_l"),
        ("eye_r", "ear_r"),
        ("ear_l", "sho_l"),
        ("ear_r", "sho_r"),
        ("sho_l", "sho_r"),
        ("sho_l", "hip_l"),
        ("sho_r", "hip_r"),
        ("hip_l", "hip_r"),
        ("sho_l", "elb_l"),
        ("elb_l", "wri_l"),
        ("sho_r", "elb_r"),
        ("elb_r", "wri_r"),
        ("hip_l", "kne_l"),
        ("kne_l", "ank_l"),
        ("hip_r", "kne_r"),
        ("kne_r", "ank_r"),
    ]

    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        threshold: float = 0.1,
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.threshold = threshold
        self.num_keypoints = 18
        self.input_height, self.input_width = 368, 432

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]
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
            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float32)

            preprocessed_data.append(padded_img)
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def rescale_keypoints(self, keypoints_list, pad, scale):
        for keypoint in keypoints_list:
            keypoint[0] = int((keypoint[0] - pad[1]) / scale)
            keypoint[1] = int((keypoint[1] - pad[0]) / scale)
        return keypoints_list

    def postprocess(self, results):
        postprocessed_detections = []
        for stages_output, img_data in zip(results[0], results[1]):
            scale, pad = img_data
            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0
            outputs = np.array(
                stages_output.getLayerFp16(stages_output.getAllLayerNames()[-1]),
            ).reshape((1, 57, 46, 54))
            for part in range(self.num_keypoints):
                probMap = outputs[0, part, :, :]
                probMap = cv2.resize(probMap, (self.input_width, self.input_height))
                keypoints = getKeypoints(probMap, self.threshold)
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)
            valid_pairs, invalid_pairs = getValidPairs(
                outputs, self.input_width, self.input_height, detected_keypoints,
            )
            personwiseKeypoints = getPersonwiseKeypoints(
                valid_pairs, invalid_pairs, keypoints_list,
            )
            image_postproc_detections = []
            keypoints_list = self.rescale_keypoints(keypoints_list, pad, scale)
            for n in range(len(personwiseKeypoints)):
                if len(personwiseKeypoints[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if personwiseKeypoints[n][kpt_id] != -1.0:
                        pose_keypoints[kpt_id, 0] = int(
                            keypoints_list[int(personwiseKeypoints[n][kpt_id]), 0],
                        )
                        pose_keypoints[kpt_id, 1] = int(
                            keypoints_list[int(personwiseKeypoints[n][kpt_id]), 1],
                        )
                pose_keypoints = np.delete(pose_keypoints, (1), axis=0)
                pose_keypoints = np.concatenate(
                    (
                        pose_keypoints,
                        np.ones((pose_keypoints.shape[0], 1), dtype=np.int32),
                    ),
                    axis=1,
                )
                pose_keypoints[np.where(pose_keypoints[:, 0] == -1)[0], :] = np.zeros(
                    3,
                )
                links = self.create_links(pose_keypoints, self.model_part_idx)
                pose = Pose(
                    score=float(personwiseKeypoints[n][18]),
                    links=links,
                    skeleton_parts=self.coco_part_labels,
                )
                image_postproc_detections.append(pose)
            postprocessed_detections.append(image_postproc_detections)
        return postprocessed_detections

    def to_device(self, device: str) -> None:
        pass

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]

    def create_links(self, skeleton, class_map):
        links = []
        for pair in self.coco_part_orders:
            joint_a_class_name = pair[0]
            joint_b_class_name = pair[1]
            link = Link(
                joint_a=Joint(
                    x=int(skeleton[class_map[joint_a_class_name]][0]),
                    y=int(skeleton[class_map[joint_a_class_name]][1]),
                    score=float(skeleton[class_map[joint_a_class_name]][2]),
                    class_name=str(joint_a_class_name),
                ),
                joint_b=Joint(
                    x=int(skeleton[class_map[joint_b_class_name]][0]),
                    y=int(skeleton[class_map[joint_b_class_name]][1]),
                    score=float(skeleton[class_map[joint_b_class_name]][2]),
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

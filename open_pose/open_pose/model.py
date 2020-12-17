import math
import os
from operator import itemgetter

import numpy as np
from modelplace_api import BaseModel, Joint, Link, Pose
from datetime import datetime, timedelta

import cv2
import depthai as dai


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True

BODY_PARTS_KPT_IDS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]
BODY_PARTS_PAF_IDS = (
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
)


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_center = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 1 : heatmap_with_borders.shape[1] - 1
    ]
    heatmap_left = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 2 : heatmap_with_borders.shape[1]
    ]
    heatmap_right = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 0 : heatmap_with_borders.shape[1] - 2
    ]
    heatmap_up = heatmap_with_borders[
        2 : heatmap_with_borders.shape[0], 1 : heatmap_with_borders.shape[1] - 1
    ]
    heatmap_down = heatmap_with_borders[
        0 : heatmap_with_borders.shape[0] - 2, 1 : heatmap_with_borders.shape[1] - 1
    ]

    heatmap_peaks = (
        (heatmap_center > heatmap_left)
        & (heatmap_center > heatmap_right)
        & (heatmap_center > heatmap_up)
        & (heatmap_center > heatmap_down)
    )
    heatmap_peaks = heatmap_peaks[
        1 : heatmap_center.shape[0] - 1, 1 : heatmap_center.shape[1] - 1
    ]
    keypoints = list(
        zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]),
    )  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(keypoints)):
            if (
                math.sqrt(
                    (keypoints[i][0] - keypoints[j][0]) ** 2
                    + (keypoints[i][1] - keypoints[j][1]) ** 2,
                )
                < 6
            ):
                suppressed[j] = 1
        keypoint_with_score_and_id = (
            keypoints[i][0],
            keypoints[i][1],
            heatmap[keypoints[i][1], keypoints[i][0]],
            total_keypoint_num + keypoint_num,
        )
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def group_keypoints(  # noqa: C901
    all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False,
):
    pose_entries = []
    all_keypoints = np.array(
        [item for sublist in all_keypoints_by_type for item in sublist],
    )
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(
                    len(pose_entries),
                ):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1  # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]  # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (
                    int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                    int(round((kpt_a[1] + kpt_b[1]) * 0.5)),
                )
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (
                    vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0]
                    + vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1]
                )

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        if not demo:
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append(
                    [kpts_a[i][3], kpts_b[j][3], cur_point_score],
                )
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [
                np.ones(pose_entry_size) * -1 for _ in range(len(connections))
            ]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = (
                    np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                )
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if (
                        pose_entries[j][kpt_a_id] == connections[i][0]
                        and pose_entries[j][kpt_b_id] == -1
                    ):
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif (
                        pose_entries[j][kpt_b_id] == connections[i][1]
                        and pose_entries[j][kpt_a_id] == -1
                    ):
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += (
                            all_keypoints[connections[i][1], 2] + connections[i][2]
                        )
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = (
                        np.sum(all_keypoints[connections[i][0:2], 2])
                        + connections[i][2]
                    )
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


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
    coco_part_idx = {b: a for a, b in enumerate(coco_part_labels)}
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
            model_description: str = "",
            **kwargs,
    ):
        super().__init__(model_path, model_name, model_description, **kwargs)
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = 18
        self.input_height, self.input_width = 256, 456

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

    def postprocess(self, results):
        postprocessed_detections = []
        for result, img_data in zip(results[0], results[1]):
            scale, pad = img_data
            stage2_heatmaps = np.array(result.getLayerFp16(result.getAllLayerNames()[-1])).reshape((1, 19, 32, 57))
            heatmaps = np.transpose(stage2_heatmaps[0], (1, 2, 0))
            heatmaps = cv2.resize(
                heatmaps,
                (0, 0),
                fx=self.upsample_ratio,
                fy=self.upsample_ratio,
                interpolation=cv2.INTER_CUBIC,
            )

            stage2_pafs = np.array(result.getLayerFp16(result.getAllLayerNames()[-2])).reshape((1, 38, 32, 57))
            pafs = np.transpose(stage2_pafs[0], (1, 2, 0))
            pafs = cv2.resize(
                pafs,
                (0, 0),
                fx=self.upsample_ratio,
                fy=self.upsample_ratio,
                interpolation=cv2.INTER_CUBIC,
            )

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(self.num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(
                    heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num,
                )

            pose_entries, all_keypoints = group_keypoints(
                all_keypoints_by_type, pafs, demo=True,
            )
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (
                    all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio
                    - pad[1]
                ) / scale

                all_keypoints[kpt_id, 1] = (
                    all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio
                    - pad[0]
                ) / scale

            image_postproc_detections = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(
                            all_keypoints[int(pose_entries[n][kpt_id]), 0],
                        )
                        pose_keypoints[kpt_id, 1] = int(
                            all_keypoints[int(pose_entries[n][kpt_id]), 1],
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
                    score=float(pose_entries[n][18]),
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
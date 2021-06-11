import cv2
import numpy as np
from modelplace_api import Joint, Link, Pose

from oak_inference_utils import DataInfo, OAKSingleStageModel, pad_img

from .utils import extract_keypoints, group_keypoints


class InferenceModel(OAKSingleStageModel):
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
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="data",
            input_shapes=(456, 256),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = 18

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
            data_infos.append(
                DataInfo(
                    scales=(scale, scale),
                    pads=tuple(pad),
                    original_width=width,
                    original_height=height,
                ),
            )

        return [preprocessed_data, data_infos]

    def postprocess(self, results):
        postprocessed_detections = []
        for result, input_info in zip(results[0], results[1]):
            (scale_x, scale_y), pads = input_info.scales, input_info.pads
            stage2_heatmaps = np.array(
                result.getLayerFp16(result.getAllLayerNames()[-1]),
            ).reshape((1, 19, 32, 57))
            heatmaps = np.transpose(stage2_heatmaps[0], (1, 2, 0))
            heatmaps = cv2.resize(
                heatmaps,
                (0, 0),
                fx=self.upsample_ratio,
                fy=self.upsample_ratio,
                interpolation=cv2.INTER_CUBIC,
            )

            stage2_pafs = np.array(
                result.getLayerFp16(result.getAllLayerNames()[-2]),
            ).reshape((1, 38, 32, 57))
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
                    - pads[1]
                ) / scale_x

                all_keypoints[kpt_id, 1] = (
                    all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio
                    - pads[0]
                ) / scale_y

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

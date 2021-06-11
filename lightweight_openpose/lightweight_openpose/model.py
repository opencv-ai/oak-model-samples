import cv2
import numpy as np
from modelplace_api import Joint, Link, Pose

from oak_inference_utils import DataInfo, OAKSingleStageModel, pad_img

from .utils import get_keypoints, get_personwise_keypoints, get_valid_pairs


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
        threshold: float = 0.1,
        model_description: str = "",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="data",
            input_shapes=(432, 368),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold
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

    @staticmethod
    def rescale_keypoints(keypoints_list, pad, scales):
        for keypoint in keypoints_list:
            keypoint[0] = int((keypoint[0] - pad[1]) / scales[0])
            keypoint[1] = int((keypoint[1] - pad[0]) / scales[1])
        return keypoints_list

    def postprocess(self, results):
        postprocessed_detections = []
        for stages_output, input_info in zip(results[0], results[1]):
            scales, pads = input_info.scales, input_info.pads
            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0
            outputs = np.array(
                stages_output.getLayerFp16(stages_output.getAllLayerNames()[-1]),
            ).reshape((1, 57, 46, 54))
            for part in range(self.num_keypoints):
                prob_map = outputs[0, part, :, :]
                prob_map = cv2.resize(prob_map, (self.input_width, self.input_height))
                keypoints = get_keypoints(prob_map, self.threshold)
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)
            valid_pairs, invalid_pairs = get_valid_pairs(
                outputs, self.input_width, self.input_height, detected_keypoints,
            )
            personwise_keypoints = get_personwise_keypoints(
                valid_pairs, invalid_pairs, keypoints_list,
            )
            image_postproc_detections = []
            keypoints_list = self.rescale_keypoints(keypoints_list, pads, scales)
            for n in range(len(personwise_keypoints)):
                if len(personwise_keypoints[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if personwise_keypoints[n][kpt_id] != -1.0:
                        pose_keypoints[kpt_id, 0] = int(
                            keypoints_list[int(personwise_keypoints[n][kpt_id]), 0],
                        )
                        pose_keypoints[kpt_id, 1] = int(
                            keypoints_list[int(personwise_keypoints[n][kpt_id]), 1],
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
                    score=float(personwise_keypoints[n][18]),
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

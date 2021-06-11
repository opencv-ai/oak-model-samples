from typing import Tuple

from modelplace_api import Joint, Link, Pose

from oak_inference_utils import OAKTwoStageModel

from .palm_processing import PalmProcessor
from .utils import *


class InferenceModel(OAKTwoStageModel):
    coco_part_labels = [
        "Wrist",
        "TMCP",
        "TPIP",
        "TDIP",
        "TTIP",
        "IMCP",
        "IPIP",
        "IDIP",
        "ITIP",
        "MMCP",
        "MPIP",
        "MDIP",
        "MTIP",
        "RMCP",
        "RPIP",
        "RDIP",
        "RTIP",
        "PMCP",
        "PPIP",
        "PDIP",
        "PTIP",
    ]
    coco_part_idx = {b: a for a, b in enumerate(coco_part_labels)}
    coco_part_orders = [
        ("Wrist", "TMCP"),
        ("TMCP", "TPIP"),
        ("TPIP", "TDIP"),
        ("TDIP", "TTIP"),
        ("Wrist", "PMCP"),
        ("PMCP", "PPIP"),
        ("PPIP", "PDIP"),
        ("PDIP", "PTIP"),
        ("Wrist", "IMCP"),
        ("IMCP", "IPIP"),
        ("IPIP", "IDIP"),
        ("IDIP", "ITIP"),
        ("Wrist", "MMCP"),
        ("MMCP", "MPIP"),
        ("MPIP", "MDIP"),
        ("MDIP", "MTIP"),
        ("Wrist", "RMCP"),
        ("RMCP", "RPIP"),
        ("RPIP", "RDIP"),
        ("RDIP", "RTIP"),
    ]

    def __init__(
        self,
        model_path: str,
        preview_shape: Tuple[int, int] = (640, 480),
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.1,
        area_threshold: float = 0.15,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="data",
            preview_shape=preview_shape,
            first_stage=PalmProcessor(threshold, preview_shape),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.output_names = ["Identity_dense/BiasAdd/Add", "Identity_1"]
        self.area_threshold = area_threshold
        self.threshold = threshold
        self.input_width, self.input_height = (
            224,
            224,
        )

    def preprocess(self, data):
        hands_bboxes = self.get_first_stage_result(data)
        preprocessed_data = []
        preprocessed_hand_regions = []
        if hands_bboxes == [[]]:
            return [preprocessed_data, preprocessed_hand_regions]
        for i, img in enumerate(data):
            img = np.array(data[i])[:, :, ::-1]
            height, width, _ = img.shape
            if not hands_bboxes[i]:
                preprocessed_data.append([])
                preprocessed_hand_regions.append([])
                continue
            # convert palm detection result to rectangle that will likely cover entire hand
            hand_regions = convert_palm_labels_to_hand_regions(
                hands_bboxes[i], height, width,
            )
            convert_hand_regions_to_rect(hand_regions, width, height)

            preprocessed_img = []
            for region in hand_regions:
                warped_image = warp_rect_img(
                    region.rect_points, img, self.input_width, self.input_height,
                )
                warped_image = warped_image.transpose((2, 0, 1))
                warped_image = warped_image[np.newaxis].astype(np.float32)
                preprocessed_img.append(warped_image)
            preprocessed_data.append(preprocessed_img)
            preprocessed_hand_regions.append(hand_regions)
        return [preprocessed_data, preprocessed_hand_regions]

    def postprocess(self, predictions):
        if not len(predictions[0]):
            return [[]]
        postprocessed_result = []
        for results, regions in zip(predictions[0], predictions[1]):
            image_predictions = []
            for result, region in zip(results, regions):
                region.lm_score = np.array(result.getLayerFp16(self.output_names[1]))
                if region.lm_score < self.threshold:
                    continue
                lm_raw = np.array(result.getLayerFp16(self.output_names[0]))
                # normalize landmarks
                landmarks = [
                    lm_raw[3 * i : 3 * (i + 1)] / self.input_width
                    for i in range(len(lm_raw) // 3)
                ]
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array(
                    [(x, y) for x, y in region.rect_points[1:]], dtype=np.float32,
                )
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(
                    np.array([(landmark[0], landmark[1]) for landmark in landmarks]),
                    axis=0,
                )
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
                image_predictions.append(
                    Pose(
                        score=region.lm_score,
                        links=self.create_links(lm_xy, self.coco_part_idx),
                        skeleton_parts=self.coco_part_labels,
                    ),
                )
            postprocessed_result.append(image_predictions)
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

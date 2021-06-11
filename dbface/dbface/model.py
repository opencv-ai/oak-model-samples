import cv2
from modelplace_api import BBox, Landmarks, Point

from oak_inference_utils import DataInfo, OAKSingleStageModel

from .utils import *


class InferenceModel(OAKSingleStageModel):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            input_name="x",
            input_shapes=(640, 480),
            model_name=model_name,
            model_description=model_description,
            **kwargs,
        )
        self.threshold = threshold

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)
            height, width, _ = img.shape
            scale_y = self.input_height / height
            scale_x = self.input_width / width
            resized_image = cv2.resize(
                img,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_CUBIC,
            )
            self.resized_image = resized_image
            resized_image = resized_image.transpose((2, 0, 1))
            resized_image = resized_image[np.newaxis].astype(np.float32)
            preprocessed_data.append(resized_image)
            data_infos.append(
                DataInfo(
                    scales=(scale_x, scale_y),
                    pads=(0, 0),
                    original_width=width,
                    original_height=height,
                ),
            )
        return [preprocessed_data, data_infos]

    def postprocess(self, predictions):
        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            scale_x, scale_y = input_info.scales
            original_h, original_w = (
                input_info.original_height,
                input_info.original_width,
            )
            hm = np.array(result.getLayerFp16(result.getAllLayerNames()[2])).reshape(
                (1, 1, 120, 160),
            )
            box = np.array(result.getLayerFp16(result.getAllLayerNames()[1])).reshape(
                (1, 4, 120, 160),
            )
            landmark = np.array(
                result.getLayerFp16(result.getAllLayerNames()[0]),
            ).reshape((1, 10, 120, 160))
            objs = detect(
                hm=hm,
                box=box,
                landmark=landmark,
                threshold=self.threshold,
                nms_iou=0.5,
            )
            image_predictions = []
            for obj in objs:
                box, confidence, landmark = obj
                image_predictions.append(
                    Landmarks(
                        bbox=BBox(
                            x1=int(np.clip(box[0] / scale_x, 0, original_w)),
                            y1=int(np.clip(box[1] / scale_y, 0, original_h)),
                            x2=int(np.clip(box[2] / scale_x, 0, original_w)),
                            y2=int(np.clip(box[3] / scale_y, 0, original_h)),
                            score=float(confidence),
                            class_name="face",
                        ),
                        keypoints=[
                            Point(
                                x=int(np.clip(x / scale_x, 0, original_w)),
                                y=int(np.clip(y / scale_y, 0, original_h)),
                            )
                            for x, y in landmark
                        ],
                    ),
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

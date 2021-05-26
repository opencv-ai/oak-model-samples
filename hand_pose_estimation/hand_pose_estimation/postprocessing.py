import csv

import numpy as np


def non_maximum_suppression(boxes, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        overlap_threshold:
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.

    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15

    boxes = boxes.astype(np.float64)

    pick = []
    x1, y1, x2, y2, confs = boxes.T

    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)

    while len(idxs) > 0:
        i = idxs[-1]

        pick.append(i)
        if len(pick) >= top_k:
            break

        idxs = idxs[:-1]

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h

        overlap = I / (area[idxs] + eps)

        idxs = idxs[overlap <= overlap_threshold]

    return pick


class Postprocessor:
    def __init__(
        self, anchors_path, nms_threshold=0.45, nms_top_k=400,
    ):
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k

        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x))

    def decode_predictions(self, predictions, output_names):
        out_clf = self._sigm(predictions[0])
        out_reg = predictions[1]
        keypoints = out_reg[:, 4:] * np.tile(self.anchors[:, 2:4], 7) / 256 + np.tile(
            self.anchors[:, 0:2], 7,
        )
        bboxes = np.empty((out_clf.shape[0], 5))
        x_center = out_reg[:, 0] / 128 + self.anchors[:, 0]
        y_center = out_reg[:, 1] / 128 + self.anchors[:, 1]
        w = out_reg[:, 2] / 128
        h = out_reg[:, 3] / 128
        bboxes[:, 0] = x_center - w / 2.0
        bboxes[:, 1] = y_center - h / 2.0
        bboxes[:, 2] = x_center + w / 2.0
        bboxes[:, 3] = y_center + h / 2.0
        bboxes[:, 4] = out_clf

        bboxes = np.clip(bboxes, 0.0, 1.0)
        idx = non_maximum_suppression(bboxes, self.nms_threshold, self.nms_top_k)
        bboxes = bboxes[idx]
        keypoints = keypoints[idx]
        kps_total = []
        for i in range(keypoints.shape[0]):
            kps = []
            for kp in range(7):
                kps.append(keypoints[i, kp * 2 : 2 + kp * 2])
            kps_total.append(kps)
        return bboxes, kps_total

import numpy as np


def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
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
    x1, y1, x2, y2 = boxes.T

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


class PriorMap(object):
    def __init__(
        self,
        image_size,
        map_size,
        minmax_size=None,
        variances=(0.1, 0.1, 0.2, 0.2),
        aspect_ratios=(1,),
        shift=None,
    ):

        self.image_size = image_size
        self.map_size = map_size
        self.minmax_size = minmax_size
        self.variances = variances
        self.aspect_ratios = aspect_ratios
        self.shift = shift

    def compute_priors(self):
        image_h, image_w = image_size = self.image_size
        map_h, map_w = map_size = self.map_size
        min_size, max_size = self.minmax_size

        step_x = image_w / map_w
        step_y = image_h / map_h
        assert (
            step_x % 1 == 0 and step_y % 1 == 0
        ), "map size %s not constiten with input size %s" % (map_size, image_size)

        linx = np.array([(0.5 + i) for i in range(map_w)]) * step_x
        liny = np.array([(0.5 + i) for i in range(map_h)]) * step_y
        box_xy = np.array(np.meshgrid(linx, liny)).reshape(2, -1).T

        shift = self.shift

        box_wh = []
        box_shift = []
        for i in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[i]
            box_wh.append([min_size * np.sqrt(ar), min_size / np.sqrt(ar)])
            box_shift.append(shift[i])

        box_wh = np.asarray(box_wh)

        box_shift = np.asarray(box_shift)
        box_shift = np.clip(box_shift, -1.0, 1.0)
        box_shift = box_shift * np.array([step_x, step_y])  # percent to pixels

        # values for individual prior boxes
        priors_shift = np.tile(box_shift, (len(box_xy), 1))
        priors_xy = np.repeat(box_xy, len(box_wh), axis=0) + priors_shift
        priors_wh = np.tile(box_wh, (len(box_xy), 1))

        priors_min_xy = priors_xy - priors_wh / 2.0
        priors_max_xy = priors_xy + priors_wh / 2.0

        priors_variances = np.tile(self.variances, (len(priors_xy), 1))

        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_variances = priors_variances
        self.priors = np.concatenate(
            [priors_min_xy, priors_max_xy, priors_variances], axis=1,
        )


class PriorUtil(object):
    """Utility for SSD prior boxes.
    """

    def __init__(self, map_sizes, aspect_ratios, shifts, scale):

        self.image_size = (256, 256)
        num_maps = len(map_sizes)

        min_dim = np.min(self.image_size)
        min_ratio = 10  # 15
        max_ratio = 100  # 90
        s = np.linspace(min_ratio, max_ratio, num_maps + 1) * min_dim / 100.0
        minmax_sizes = [(round(s[i]), round(s[i + 1])) for i in range(len(s) - 1)]

        minmax_sizes = np.array(minmax_sizes) * scale

        self.prior_maps = []
        for i, map_size in enumerate(map_sizes):
            m = PriorMap(
                image_size=self.image_size,
                map_size=map_size,
                minmax_size=minmax_sizes[i],
                variances=[0.1, 0.1, 0.2, 0.2],
                aspect_ratios=aspect_ratios,
                shift=shifts,
            )
            self.prior_maps.append(m)
        self.update_priors()

        self.nms_top_k = 400
        self.nms_thresh = 0.45

    def update_priors(self):
        priors_xy = []
        priors_wh = []
        priors_variances = []

        map_offsets = [0]
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]

            # compute prior boxes
            m.compute_priors()

            # collect prior data
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors_variances.append(m.priors_variances)
            map_offsets.append(map_offsets[-1] + len(m.priors))

        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors_variances = np.concatenate(priors_variances, axis=0)

    def decode_results(self, model_output, confidence_threshold=0.7, keep_top_k=200):

        prior_mask = model_output[:, 17:] > confidence_threshold

        mask = np.any(prior_mask[:, 1:], axis=1)
        prior_mask = prior_mask[mask]
        mask = np.ix_(mask)[0]
        model_output = model_output[mask]
        priors_xy = self.priors_xy[mask] / self.image_size
        priors_wh = self.priors_wh[mask] / self.image_size
        priors_variances = self.priors_variances[mask, :]
        priors_xy_minmax = np.hstack(
            [priors_xy - priors_wh / 2, priors_xy + priors_wh / 2],
        )

        offsets = model_output[:, 13:17]
        offsets_quads = model_output[:, 5:13]
        confidence = model_output[:, 17:]

        ref = priors_xy_minmax[:, (0, 1, 2, 1, 2, 3, 0, 3)]  # corner points
        variances_xy = priors_variances[:, 0:2]

        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:, 0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:, 2:4])
        boxes[:, 0:2] = boxes_xy - boxes_wh / 2.0  # xmin, ymin
        boxes[:, 2:4] = boxes_xy + boxes_wh / 2.0  # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)

        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:, c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]

                idx = non_maximum_suppression(
                    boxes_to_process, confs_to_process, self.nms_thresh, self.nms_top_k,
                )

                good_quads = ref[mask][idx] + offsets_quads[mask][idx] * np.tile(
                    priors_wh[mask][idx] * variances_xy[mask][idx], (1, 4),
                )

                results.extend(good_quads)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 5])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0, 6))

        return results

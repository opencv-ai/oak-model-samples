import numpy as np
from numpy.lib.stride_tricks import as_strided


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def calculate_iou(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def _exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [_exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([_exp(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj[1], reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and calculate_iou(obj[0], objs[j][0]) > iou:
                flags[j] = 1
    return keep


def max_pooling(x, kernel_size, stride=1, padding=1):
    x = np.pad(x, padding, mode="constant")
    output_shape = (
        (x.shape[0] - kernel_size) // stride + 1,
        (x.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    x_w = as_strided(
        x,
        shape=output_shape + kernel_size,
        strides=(stride * x.strides[0], stride * x.strides[1]) + x.strides,
    )
    x_w = x_w.reshape(-1, *kernel_size)

    return x_w.max(axis=(1, 2)).reshape(output_shape)


def detect(hm, box, landmark, threshold=0.4, nms_iou=0.5):
    hm_pool = max_pooling(hm[0, 0, :, :], 3, 1, 1)  # 1,1,240,320
    interest_points = (hm == hm_pool) * hm  # screen out low-conf pixels
    flat = interest_points.ravel()  # flatten
    indices = np.argsort(flat)[::-1]  # index sort
    scores = np.array([flat[idx] for idx in indices])

    hm_height, hm_width = hm.shape[2:]
    ys = indices // hm_width
    xs = indices % hm_width
    box = box.reshape(box.shape[1:])  # 4,240,320
    landmark = landmark.reshape(landmark.shape[1:])  # 10,240,,320

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break
        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (_exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append([xyrb, score, box_landmark])
    return nms(objs, iou=nms_iou)

import numpy as np


def rotated_rectangle(rotatedRect):
    x, y, width, height, angle = rotatedRect

    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))

    t = np.array(
        [
            [np.cos(angle), -np.sin(angle), x - x * np.cos(angle) + y * np.sin(angle)],
            [np.sin(angle), np.cos(angle), y - x * np.sin(angle) - y * np.cos(angle)],
            [0, 0, 1],
        ],
    )

    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))

    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))

    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))

    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))

    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

    return points


def non_max_suppression(boxes, probs=None, overlapThresh=0.4):
    # if there are no boxes, return an empty list
    if boxes.size == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    # grab the coordinates of the bounding boxes
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])),
        )

    # return only the bounding boxes that were picked
    return boxes[pick]


def decode_predictions(scores, geometry1, geometry2, threshold, scale_w, scale_h):

    scores = scores.squeeze()
    valid_detections = scores > threshold
    anchor_grid = 4 * np.indices(scores.shape)
    scores = scores[valid_detections]
    offsets_top = geometry1[0, 0, valid_detections]
    offsets_right = geometry1[0, 1, valid_detections]
    offsets_bottom = geometry1[0, 2, valid_detections]
    offsets_left = geometry1[0, 3, valid_detections]
    top_left_y = anchor_grid[0, valid_detections] - offsets_top
    top_left_x = anchor_grid[1, valid_detections] - offsets_left
    h = offsets_top + offsets_bottom
    w = offsets_right + offsets_left
    center_y = (top_left_y + h / 2) / scale_h
    center_x = (top_left_x + w / 2) / scale_w
    h, w = h / scale_h, w / scale_w
    angles = geometry2.squeeze()[valid_detections]

    rects = np.dstack((center_x, center_y, w, h, -angles)).squeeze()
    if len(rects.shape) == 1:
        rects = rects[np.newaxis]

    return (rects, scores)

from copy import deepcopy

import numpy as np


def box_iou(bboxes1: np.ndarray, bboxes2: np.ndarray, box_format: str = 'xywh', do_ioa: bool = False):
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'xyxy').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
    """
    bboxes1 = deepcopy(bboxes1)
    bboxes2 = deepcopy(bboxes2)
    if box_format == 'xywh':
        # layout: (x0, y0, w, h)
        bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
        bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
        bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
        bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
    elif box_format == 'xyxy':
        bboxes1 = bboxes1
        bboxes2 = bboxes2
    else:
        raise Exception(f'box_format {box_format} is not implemented')

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    if do_ioa:
        ioas = np.zeros_like(intersection)
        valid_mask = area1 > 0 + np.finfo('float').eps
        ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

        return ioas
    else:
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
        intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
        intersection[union <= 0 + np.finfo('float').eps] = 0
        union[union <= 0 + np.finfo('float').eps] = 1
        ious = intersection / union
        return ious

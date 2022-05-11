from typing import Dict, List

import numpy as np
from cv_eval_metrics.objects import TrackingObject
from cv_eval_metrics.utils import box_iou


class TMetricConfig:
    @property
    def pred_dets(self) -> List[np.ndarray]:
        return self.__pred_dets

    @property
    def pred_ids(self) -> List[np.ndarray]:
        return self.__pred_ids

    @property
    def pred_dets_cnt(self) -> int:
        return self.__pred_dets_cnt

    @property
    def pred_ids_cnt(self) -> int:
        return self.__pred_ids_cnt

    @property
    def gt_dets(self) -> List[np.ndarray]:
        return self.__gt_dets

    @property
    def gt_ids(self) -> List[np.ndarray]:
        return self.__gt_ids

    @property
    def gt_dets_cnt(self) -> int:
        return self.__gt_dets_cnt

    @property
    def gt_ids_cnt(self) -> int:
        return self.__gt_ids_cnt

    @property
    def timestamps_cnt(self) -> int:
        return len(self.__gt_dets)

    @property
    def threshold(self) -> float:
        return self.__threshold

    @property
    def classes(self) -> list:
        return self.__classes

    @property
    def similarity_scores(self) -> List:
        return self.__similarity_scores

    def __init__(self, classes: list = None, threshold: float = 0.5, box_format: str = 'xywh') -> None:
        self.__pred_ids: List = []
        self.__pred_dets: List = []
        self.__pred_ids_cnt: int = 0
        self.__pred_dets_cnt: int = 0

        self.__gt_dets: List = []
        self.__gt_ids: List = []
        self.__gt_ids_cnt: int = 0
        self.__gt_dets_cnt: int = 0

        self.__threshold = threshold
        self.__classes = classes

        self.__box_format = box_format

        self.__similarity_scores: List = []

    def update(self, gt: TrackingObject, pred: TrackingObject):
        self.__similarity_scores: List = []

        self.__gt_dets = gt.dets
        self.__gt_ids = gt.ids
        self.__gt_dets_cnt = gt.num_dets
        self.__gt_ids_cnt = gt.num_ids

        self.__pred_dets = pred.dets
        self.__pred_ids = pred.ids
        self.__pred_dets_cnt = pred.num_dets
        self.__pred_ids_cnt = pred.num_ids

        for timestamp, (gt_dets_t, pred_dets_t) in enumerate(zip(self.__gt_dets, self.__pred_dets)):
            ious = box_iou(pred_dets_t, gt_dets_t, self.__box_format)
            self.__similarity_scores.append(ious)

import os
from typing import List, Optional, Union

import numpy as np
from cv_eval_metrics.objects import DetectionObject
from cv_eval_metrics.utils import bboxes


class DMetricConfig:
    @property
    def classes(self) -> list:
        return self.__classes

    @property
    def bbox_format(self) -> str:
        return self.__bbox_format

    @property
    def iou_thresholds(self) -> List[float]:
        return self.__iou_thresholds

    @property
    def rec_thresholds(self) -> List[float]:
        return self.__rec_thresholds

    @property
    def max_dets(self) -> List[int]:
        return self.__max_dets

    @property
    def area_scales(self) -> dict:
        return self.__area_scales

    @property
    def pred_bboxes(self) -> np.ndarray:
        return self.__pred_bboxes

    @property
    def pred_labels(self) -> np.ndarray:
        return self.__pred_labels

    @property
    def pred_scores(self) -> np.ndarray:
        return self.__pred_scores

    @property
    def gt_bboxes(self) -> np.ndarray:
        return self.__gt_bboxes

    @property
    def gt_labels(self) -> np.ndarray:
        return self.__gt_labels

    def __init__(
        self,
        classes: Union[List[str], str],
        bbox_format: str = "xyxy",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
    ) -> None:
        self.__bbox_format = bbox_format
        self.__iou_thresholds = iou_thresholds or np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.__rec_thresholds = rec_thresholds or np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.__max_dets: list = [1, 10, 100]
        self.__area_scales: dict = {
            'all': (0 ** 2, 1e5 ** 2),
            'small': (0 ** 2, 32 ** 2),
            'medium': (32 ** 2, 96 ** 2),
            'large': (96 ** 2, 1e5 ** 2)
        }

        if isinstance(classes, list):
            self.__classes = classes
        elif isinstance(classes, str):
            if not os.path.isfile(classes):
                raise ValueError(f"{classes} label file does not found!")

            if os.path.splitext(classes)[-1] != ".txt":
                raise ValueError(f"Currently support only text file with classes line by line.")

            with open(classes, 'r') as f:
                self.__classes = f.read().splitlines()
        else:
            raise TypeError("Unsupported Type!")

        self.__pred_bboxes: np.ndarray = None
        self.__pred_scores: np.ndarray = None
        self.__pred_labels: np.ndarray = None

        self.__gt_bboxes: np.ndarray = None
        self.__gt_labels: np.ndarray = None

    def update(self, gt: DetectionObject, pred: DetectionObject):
        self.__pred_bboxes = pred.bboxes
        self.__pred_labels = pred.labels
        self.__pred_scores = pred.scores

        self.__gt_bboxes = gt.bboxes
        self.__gt_labels = gt.labels

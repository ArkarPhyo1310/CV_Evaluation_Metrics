import os
from typing import Dict, List, Union

from cv_eval_metrics.objects import ClassificationObject


class CMetricConfig:
    @property
    def pred_names(self) -> List[str]:
        return self.__pred_names

    @property
    def pred_labels(self) -> List[int]:
        return self.__pred_labels

    @property
    def pred_scores_list(self) -> List:
        return self.__pred_scores_list

    @property
    def gt_names(self) -> List[str]:
        return self.__gt_names

    @property
    def gt_labels(self) -> List[int]:
        return self.__gt_labels

    @property
    def classes(self) -> Dict[int, str]:
        return self.__classes

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def average(self) -> str:
        return self.__average

    @property
    def top_k(self) -> int:
        return self.__top_k

    @property
    def mdmc(self) -> str:
        return self.__mdmc

    @property
    def multi_label(self) -> bool:
        return self.__multi_label

    @property
    def normalize(self) -> str:
        return self.__normalize

    def __init__(
        self,
        classes: Union[List[str], str],
        average: str = "micro", mdmc: str = "global",
        multi_label: bool = False, normalize: str = None,
        top_k: int = None
    ) -> None:
        self.__average = average
        self.__top_k = top_k
        self.__mdmc = mdmc
        self.__multi_label = multi_label
        self.__normalize = normalize

        self.__classes: list = []
        self.__pred_names: List[str] = []
        self.__pred_labels: List[int] = []
        self.__pred_scores_list: List = []

        self.__gt_names: List[str] = []
        self.__gt_labels: List[int] = []

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

        if normalize is not None and normalize not in ["pred", "true", "all"]:
            raise ValueError("Confusion Matrix Normalization value is invalid! Must be 'pred', 'true', 'all' or None")

        self.__num_classes = len(self.__classes)

    def update(self, gt: ClassificationObject, pred: ClassificationObject):
        self.__gt_names = gt.file_names
        self.__gt_labels = gt.labels

        self.__pred_names = pred.file_names
        self.__pred_labels = pred.labels

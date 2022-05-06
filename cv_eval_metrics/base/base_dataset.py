import csv
import os
from abc import ABC, abstractmethod
from glob import glob
from typing import Union

from cv_eval_metrics.config import CMetricConfig, TMetricConfig


class BaseDataset(ABC):
    def __init__(self, gt_path: str, pred_path: str, file_format: str) -> None:
        self._gt_path: str = gt_path
        self._pred_path: str = pred_path
        self._file_format: str = file_format
        self._valid_labels = [1]

    @abstractmethod
    def process(self, gt_file: str, pred_file: str):
        ...

    @abstractmethod
    def assign(self, metric_cfg: Union[TMetricConfig, CMetricConfig]):
        ...

    def _check_path(self):
        gt_files = []
        pred_files = []

        if os.path.isdir(self._gt_path):
            gt_files = glob(os.path.join(self._gt_path, f"*.{self._file_format}"))
        elif os.path.isfile(self._gt_path):
            gt_files = [self._gt_path]
        else:
            raise Exception(f"{self._gt_path} is not a valid path!")

        if os.path.isdir(self._pred_path):
            pred_files = glob(os.path.join(self._pred_path, f"*.{self._file_format}"))
        elif os.path.isfile(self._pred_path):
            pred_files = [self._pred_path]
        else:
            raise Exception(f"{self._pred_path} is not a valid path!")

        if len(gt_files) != len(pred_files):
            raise Exception("Ground Truth and Prediction fodler have different files!")

        if len(gt_files) == 0 or len(pred_files) == 0:
            raise Exception("No Groundtruth or Predictions data are found!")

        return gt_files, pred_files

    def _check_header(self, file):
        with open(file, 'r') as csvfile:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(csvfile.read(2048))

        return has_header

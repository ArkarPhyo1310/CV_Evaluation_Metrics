from typing import Dict, List

import numpy as np

from cv_eval_metrics.abstract import BaseMetric
from cv_eval_metrics.config import CMetricConfig
from cv_eval_metrics.utils.classification_utils import input_format_classification, prob_to_cls


class ConfusionMatrix(BaseMetric):
    @property
    def metric_fields(self) -> List[str]:
        return self._metric_fields

    def __init__(self) -> None:
        super().__init__()
        self._metric_fields = ["Confusion Matrix"]

    def compute(self, cfg: CMetricConfig) -> Dict:
        if cfg.normalize not in ["pred", "true", "all"]:
            raise ValueError("Confusion Matrix Normalization value is invalid! Must be 'pred', 'true' or 'all'")
        num_classes = cfg.num_classes
        multi_label = cfg.multi_label
        cm = np.zeros((num_classes, num_classes))
        metric_dict = {}
        for field in self._metric_fields:
            metric_dict[field] = cm

        gt = input_format_classification(cfg.classes, cfg.gt_labels)
        pred = input_format_classification(cfg.classes, cfg.pred_labels)

        if pred.dtype == np.float32:
            pred = prob_to_cls(pred)

        for gt, pred in zip(gt, pred):
            metric_dict[field][gt, pred] += 1

        if cfg.normalize == "pred":
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif cfg.normalize == "true":
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif cfg.normalize == "sum":
            cm = cm.astype('float') / cm.sum()
        else:
            cm = cm.astype('float')

        metric_dict["Confusion Matrix"] = cm
        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        return all_res

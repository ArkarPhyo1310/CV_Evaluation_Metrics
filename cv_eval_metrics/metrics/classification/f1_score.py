from typing import Any, Dict, List, Union

import numpy as np
from cv_eval_metrics.base import BaseMetric
from cv_eval_metrics.config import CMetricConfig
from cv_eval_metrics.utils.classification_utils import (
    calculate_base, choose_topk, input_format_classification, to_onehot)


class F1Score(BaseMetric):
    @property
    def metric_fields(self) -> List[str]:
        return self._metric_fields

    def __init__(self) -> None:
        super().__init__()
        self._metric_fields = ["F1-Score"]
        self.__metric_supp_list = ["TP", "FP", "TN", "FN"]

    def compute(self, cfg: CMetricConfig) -> Dict:
        self._check_classification_attr(cfg)
        self.top_k = cfg.top_k

        metric_dict = {}

        for field in self._metric_fields + self.__metric_supp_list:
            metric_dict[field] = 0

        gt = input_format_classification(cfg.classes, cfg.gt_labels)
        pred = input_format_classification(cfg.classes, cfg.pred_labels)

        if self.top_k is not None:
            pred = choose_topk(pred, self.top_k, cfg.multi_label)
        else:
            pred = to_onehot(pred, cfg.num_classes)

        gt = to_onehot(gt, cfg.num_classes)

        tp, fp, tn, fn = calculate_base(
            pred,
            gt,
            cfg.multi_label,
            cfg.mdmc,
            cfg.average
        )

        metric_dict["TP"] = tp
        metric_dict["FP"] = fp
        metric_dict["TN"] = tn
        metric_dict["FN"] = fn

        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        """Combines metrics across all sequences"""
        metric_dict = {}
        for field in self.__metric_supp_list:
            metric_dict[field] = self._combine_sum(all_res, field)
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    def _compute_final_fields(self, metric_dict: Dict):
        metric_dict["Precision"] = np.nan_to_num((metric_dict["TP"])/(metric_dict["TP"]+metric_dict["FP"]))
        metric_dict["Recall"] = np.nan_to_num((metric_dict["TP"])/(metric_dict["TP"]+metric_dict["FN"]))
        metric_dict["F1-Score"] = 2 * np.nan_to_num((metric_dict["Precision"]
                                                    * metric_dict["Recall"]) / (metric_dict["Precision"] + metric_dict["Recall"]))
        return metric_dict

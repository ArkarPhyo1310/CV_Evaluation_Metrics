from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod

from tabulate import tabulate

from cv_evaluation_metrics.config.t_metric_cfg import TMetricConfig


class TBaseMetric(ABC):
    def __init__(self) -> None:
        self.metric_list: List[str] = []

    @abstractmethod
    def run_eval(self, cfg: TMetricConfig):
        ...

    @abstractmethod
    def combine_sequence(self, all_res: Dict):
        ...

    # @abstractmethod
    # def combine_classes_class_averaged(self, all_res: Dict, ignore_empyt_classes: bool = False):
    #     ...

    # @abstractmethod
    # def combine_classes_det_averaged(self, all_res: Dict):
    #     ...

    @classmethod
    def get_name(cls):
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res: Dict, field: str):
        return sum([all_res[k][field] for k in all_res.keys()])

    @staticmethod
    def _combine_weighted_avg(all_res, field, comb_res, weight_field):
        return sum([all_res[k][field] * all_res[k][weight_field] for k in all_res.keys()]) / np.maximum(1.0,
                                                                                                        comb_res[weight_field])

    def record_table(self, table_res: Dict, tracker_name: str, cls_name: str):
        all_res = []
        metric_name = self.get_name()
        for seq, results in sorted(table_res.items()):
            if seq == "COMBINED_SEQ":
                continue

            summary_res = self._summary_row(results)
            seq_res = [seq] + summary_res
            all_res.append(seq_res)

        summary_res = ['COMBINED'] + self._summary_row(table_res["COMBINED_SEQ"])

        all_res.append(summary_res)
        header = [metric_name + ': ' + tracker_name + '-' + cls_name] + self.metric_list

        return all_res, header

    def _summary_row(self, results_):
        vals = []
        for h in self.metric_list:
            if isinstance(results_[h], np.ndarray):
                vals.append("{0:1.4g}".format(100 * np.mean(results_[h])))
            elif isinstance(results_[h], float):
                vals.append("{0:1.4g}".format(100 * float(results_[h])))
            elif isinstance(results_[h], int) or isinstance(results_[h], np.int64):
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this field type.")
        return vals

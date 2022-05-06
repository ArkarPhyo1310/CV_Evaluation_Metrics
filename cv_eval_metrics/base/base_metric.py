from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
from cv_eval_metrics.config import CMetricConfig, TMetricConfig

np.seterr(divide='ignore', invalid='ignore')


class BaseMetric(ABC):
    @property
    def metric_fields(self) -> List:
        return self._metric_fields

    def __init__(self) -> None:
        self._metric_fields: List[str] = []

    @abstractmethod
    def compute(self, cfg: Union[TMetricConfig, CMetricConfig]) -> Dict:
        ...

    @abstractmethod
    def combine_result(self, all_res: Dict) -> Dict:
        ...

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res: Dict, field: str):
        return sum([all_res[k][field] for k in all_res.keys()])

    @staticmethod
    def _combine_weighted_avg(all_res, field, comb_res, weight_field):
        return sum([all_res[k][field] * all_res[k][weight_field] for k in all_res.keys()]) / np.maximum(1.0,
                                                                                                        comb_res[weight_field])

    def record_table(self, table_res: Dict, model_name: str, show_overall: bool = True) -> Tuple[List, List]:
        all_res = []
        metric_name = self.get_name()
        for seq, results in sorted(table_res.items()):
            if seq == "COMBINED_SEQ":
                continue

            summary_res = self._summary_row(results)
            seq_res = [seq] + summary_res
            all_res.append(seq_res)

        if show_overall:
            summary_res = ['COMBINED'] + self._summary_row(table_res["COMBINED_SEQ"])
            all_res.append(summary_res)

        header = [metric_name + ': ' + model_name] + self._metric_fields

        return all_res, header

    def _summary_row(self, results_: Dict) -> List:
        vals = []
        for h in self.metric_fields:
            if h == "Confusion Matrix":
                vals.append(str(results_[h].tolist()))
            elif isinstance(results_[h], np.ndarray):
                vals.append("{0:1.4g}".format(100 * np.mean(results_[h])))
            elif isinstance(results_[h], float):
                vals.append("{0:1.4g}".format(100 * float(results_[h])))
            elif isinstance(results_[h], int) or isinstance(results_[h], np.int64):
                vals.append("{0:d}".format(int(results_[h])))
            else:
                raise NotImplementedError("Summary function not implemented for this field type.")
        return vals

    def _check_classification_attr(self, cfg: CMetricConfig) -> None:
        if cfg.average not in ["micro", "macro"]:
            raise ValueError(f"{cfg.average} must be either 'micro' or 'macro'!")

        if cfg.mdmc not in ["global", "samplewise"]:
            raise ValueError(f"{cfg.average} must be either 'global' or 'samplewise'!")

        if cfg.top_k is not None:
            if cfg.top_k > cfg.num_classes:
                raise ValueError("top_k value must be lower than num_classes!")

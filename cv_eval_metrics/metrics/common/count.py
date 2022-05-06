from typing import Dict, List, Union

from cv_eval_metrics.config import TMetricConfig, CMetricConfig
from cv_eval_metrics.base import BaseMetric


class COUNT(BaseMetric):
    """Class which simply counts the number of pred and gt detections and ids"""
    @property
    def metric_fields(self) -> List[str]:
        return self._metric_fields

    def __init__(self, task: str = "tracking") -> None:
        super().__init__()
        self.task = task
        self._metric_fields = ['Pred_Cnts', 'GT_Cnts']

    def compute(self, cfg: Union[TMetricConfig, CMetricConfig]) -> Dict:
        if self.task == "tracking":
            self._metric_fields = ['Pred_Dets', 'GT_Dets', 'Pred_IDs', 'GT_IDs']
            metric_dict = {
                'Pred_Dets': cfg.pred_dets_cnt,
                'GT_Dets': cfg.gt_dets_cnt,
                'Pred_IDs': cfg.pred_ids_cnt,
                'GT_IDs': cfg.gt_ids_cnt,
                'Frames': cfg.timestamps_cnt,
                'Pred_Cnts': cfg.pred_dets_cnt
            }
        elif self.task == "classification":
            metric_dict = {
                'Pred_Cnts': len(cfg.pred_labels),
                'GT_Cnts': len(cfg.gt_labels)
            }
        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        metric_dict = {}
        for field in self.metric_fields:
            metric_dict[field] = self._combine_sum(all_res, field)

        return metric_dict

from typing import Dict

from cv_evaluation_metrics.config import TMetricConfig
from cv_evaluation_metrics.tracking.metrics import TBaseMetric


class COUNT(TBaseMetric):
    """Class which simply counts the number of pred and gt detections and ids"""

    def __init__(self) -> None:
        super().__init__()
        self.metric_list = ['Pred_Dets', 'GT_Dets', 'Pred_IDs', 'GT_IDs']

    def run_eval(self, cfg: TMetricConfig):
        metric_dict = {
            'Pred_Dets': cfg.pred_dets_cnt,
            'GT_Dets': cfg.gt_dets_cnt,
            'Pred_IDs': cfg.pred_ids_cnt,
            'GT_IDs': cfg.gt_ids_cnt,
            'Frames': cfg.timestamps_cnt
        }
        return metric_dict

    def combine_sequence(self, all_res: Dict):
        metric_dict = {}
        for field in self.metric_list:
            metric_dict[field] = self._combine_sum(all_res, field)

        return metric_dict

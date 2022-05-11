from typing import Dict, List

import numpy as np
from cv_eval_metrics.base import BaseMetric
from cv_eval_metrics.config import TMetricConfig
from scipy.optimize import linear_sum_assignment


class IDENTITY(BaseMetric):
    """Class which implements ID metrics"""
    @property
    def metric_fields(self) -> List:
        return self._metric_fields

    def __init__(self) -> None:
        super().__init__()
        self._metric_fields = ['IDF1', 'IDRecall', 'IDPrecision']
        self.__metric_supp_list = ['IDTP', 'IDFN', 'IDFP']

    def compute(self, cfg: TMetricConfig) -> Dict:
        metric_dict = {}
        for field in self.metric_fields + self.__metric_supp_list:
            metric_dict[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if cfg.pred_dets_cnt == 0:
            metric_dict['IDFN'] = cfg.gt_dets_cnt
            return metric_dict

        if cfg.gt_dets_cnt == 0:
            metric_dict['IDFP'] = cfg.pred_dets_cnt
            return metric_dict

        threshold = cfg.threshold

        # Variables counting global association
        potential_matches_count = np.zeros((cfg.pred_ids_cnt, cfg.gt_ids_cnt))
        gt_id_count = np.zeros(cfg.gt_ids_cnt)
        pred_id_count = np.zeros(cfg.pred_ids_cnt)

        for timestamp, (gt_ids_t, pred_ids_t) in enumerate(zip(cfg.gt_ids, cfg.pred_ids)):
            # Count the potential matches between ids in each timestamp
            matches_mask = np.greater_equal(cfg.similarity_scores[timestamp], threshold)
            match_idx_pred, match_idx_gt = np.nonzero(matches_mask)
            potential_matches_count[pred_ids_t[match_idx_pred], gt_ids_t[match_idx_gt]] += 1

            # Calculate the total number of dets for each git_id and pred_id
            gt_id_count[gt_ids_t] += 1
            pred_id_count[pred_ids_t] += 1

        # Calculate optimal assignment cost matrix for ID metrics
        num_gt_ids = cfg.gt_ids_cnt
        num_pred_ids = cfg.pred_ids_cnt
        fp_matrix = np.zeros((num_gt_ids + num_pred_ids, num_gt_ids + num_pred_ids))
        fn_matrix = np.zeros((num_gt_ids + num_pred_ids, num_gt_ids + num_pred_ids))

        fp_matrix[:num_pred_ids, num_gt_ids:] = 1e10
        fn_matrix[num_pred_ids:, :num_gt_ids] = 1e10

        for gt_id in range(num_gt_ids):
            fn_matrix[:num_pred_ids, gt_id] = gt_id_count[gt_id]
            fn_matrix[num_pred_ids + gt_id, gt_id] = gt_id_count[gt_id]

        for pred_id in range(num_pred_ids):
            fp_matrix[pred_id, :num_gt_ids] = pred_id_count[pred_id]
            fp_matrix[pred_id, pred_id + num_gt_ids] = pred_id_count[pred_id]

        fn_matrix[:num_pred_ids, :num_gt_ids] -= potential_matches_count
        fp_matrix[:num_pred_ids, :num_gt_ids] -= potential_matches_count

        # Hungarian Algorithm
        matched_rows, matched_cols = linear_sum_assignment(fn_matrix + fp_matrix)

        # Accumulate basic statictis
        metric_dict['IDFN'] = fn_matrix[matched_rows, matched_cols].sum().astype(np.int)
        metric_dict['IDFP'] = fp_matrix[matched_rows, matched_cols].sum().astype(np.int)
        metric_dict['IDTP'] = (gt_id_count.sum() - metric_dict['IDFN']).astype(np.int)

        # Calculate Final ID Scores
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        """Combines metrics across all sequences"""
        metric_dict = {}
        for field in self.__metric_supp_list:
            metric_dict[field] = self._combine_sum(all_res, field)
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    @staticmethod
    def _compute_final_fields(metric_dict: Dict) -> Dict:
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        metric_dict['IDRecall'] = metric_dict['IDTP'] / np.maximum(1.0, metric_dict['IDTP'] + metric_dict['IDFN'])
        metric_dict['IDPrecision'] = metric_dict['IDTP'] / np.maximum(1.0, metric_dict['IDTP'] + metric_dict['IDFP'])
        metric_dict['IDF1'] = metric_dict['IDTP'] / np.maximum(
            1.0, metric_dict['IDTP'] + 0.5 * metric_dict['IDFP'] + 0.5 * metric_dict['IDFN'])
        return metric_dict

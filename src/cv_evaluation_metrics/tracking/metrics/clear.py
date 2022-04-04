from typing import Dict, List, Union

import numpy as np
from cv_evaluation_metrics.config import TMetricConfig
from cv_evaluation_metrics.tracking.metrics import TBaseMetric
from cv_evaluation_metrics.utils import box_iou
from scipy.optimize import linear_sum_assignment


class CLEAR(TBaseMetric):
    """Class which implements the CLEAR metrics"""

    def __init__(self):
        super().__init__()
        self.metric_list: List[str] = ['MOTA', 'MOTP', 'MODA', 'Recall', 'Precision',
                                       'MTR', 'PTR', 'MLR', 'sMOTA', 'TP',
                                       'FN', 'FP', 'IDSW', 'MT', 'PT',
                                       'ML', 'FRAG']

        self.extra_list: List[str] = ['FRAMES', 'F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.integer_list: List[str] = [
            'TP', 'FN', 'FP', 'IDSW', 'MT', 'PT', 'ML', 'FRAG', 'FRAMES', 'MOTP_sum'
        ]

    def run_eval(self, cfg: TMetricConfig):
        metric_dict: Dict[str, Union[int, float]] = {}
        for field in self.metric_list + self.extra_list:
            metric_dict[field] = 0

        threshold = cfg.threshold
        num_gt_ids = cfg.gt_ids_cnt
        num_timestamps = cfg.timestamps_cnt

        if cfg.pred_dets_cnt == 0:
            metric_dict['FN'] = cfg.gt_dets_cnt
            metric_dict['ML'] = cfg.gt_ids_cnt
            metric_dict['MLR'] = 1.0
            return metric_dict

        if cfg.gt_dets_cnt == 0:
            metric_dict['FP'] = cfg.pred_dets_cnt
            metric_dict['MLR'] = 1.0
            return metric_dict

        # For MT/ML/PT/FRAG - (mostly_tracked, mostly_lost, partial_tracked, Number of Fragmentations)
        gt_id_count = np.zeros(num_gt_ids)
        gt_matched_count = np.zeros(num_gt_ids)
        gt_frag_count = np.zeros(num_gt_ids)
        # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
        # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
        prev_pred_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
        prev_timestamp_pred_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

        if len(cfg.similarity_scores) == 0:
            for timestamp, (gt_dets_t, pred_dets_t) in enumerate(zip(cfg.gt_dets, cfg.pred_dets)):
                ious = box_iou(gt_dets_t, pred_dets_t)
                cfg.similarity_scores.append(ious)

        # Calculate scores for each timestamp
        for timestamp, (gt_ids_t, pred_ids_t) in enumerate(zip(cfg.gt_ids, cfg.pred_ids)):
            # Deal with the case that there are no gt_det/pred_det in a timestamp
            if len(gt_ids_t) == 0:
                metric_dict['FP'] += len(pred_ids_t)
                continue

            if len(pred_ids_t) == 0:
                metric_dict['FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            # Calculate score matrix to first minimise IDSWs from previous frame, then maximise MOTP secondarily
            similarity = cfg.similarity_scores[timestamp]
            score_matrix = (pred_ids_t[np.newaxis, :] == prev_timestamp_pred_id[gt_ids_t[:, np.newaxis]])
            score_matrix = 1000 * score_matrix + similarity
            score_matrix[similarity < threshold - np.finfo('float').eps] = 0

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_matrix)
            actually_matched_mask = score_matrix[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_pred_ids = pred_ids_t[match_cols]

            # Calculate IDSW for MOTA
            prev_matched_pred_ids = prev_pred_id[matched_gt_ids]
            is_idsw = (np.logical_not(np.isnan(prev_matched_pred_ids))) & (
                np.not_equal(matched_pred_ids, prev_matched_pred_ids))

            metric_dict['IDSW'] += np.sum(is_idsw)

            # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestamp
            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestamp_pred_id)
            prev_pred_id[matched_gt_ids] = matched_pred_ids
            prev_timestamp_pred_id[:] = np.nan
            prev_timestamp_pred_id[matched_gt_ids] = matched_pred_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestamp_pred_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            # Calculate and accumulate basic statistics
            num_matches = len(matched_gt_ids)
            metric_dict['TP'] += num_matches
            metric_dict['FN'] += len(gt_ids_t) - num_matches
            metric_dict['FP'] += len(pred_ids_t) - num_matches
            if num_matches > 0:
                metric_dict['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        # Calculate MT/ML/PT/Frag/MOTP
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        metric_dict['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        metric_dict['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - metric_dict['MT']
        metric_dict['ML'] = num_gt_ids - metric_dict['MT'] - metric_dict['PT']
        metric_dict['FRAG'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1), dtype=np.int64)
        metric_dict['MOTP'] = metric_dict['MOTP_sum'] / np.maximum(1.0, metric_dict['TP'])

        metric_dict['FRAMES'] = num_timestamps

        # Calculate final CLEAR scores
        metric_dict = self._compute_final_fields(metric_dict)
        cfg.similarity_scores = []
        return metric_dict

    def combine_sequence(self, all_res: Dict):
        """Combines metrics across all sequences"""
        metric_dict = {}
        for field in self.integer_list:
            metric_dict[field] = self._combine_sum(all_res, field)
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    @staticmethod
    def _compute_final_fields(metric_dict: Dict):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = metric_dict['MT'] + metric_dict['ML'] + metric_dict['PT']
        metric_dict['MTR'] = metric_dict['MT'] / np.maximum(1.0, num_gt_ids)
        metric_dict['MLR'] = metric_dict['ML'] / np.maximum(1.0, num_gt_ids)
        metric_dict['PTR'] = metric_dict['PT'] / np.maximum(1.0, num_gt_ids)
        metric_dict['Recall'] = metric_dict['TP'] / np.maximum(1.0, metric_dict['TP'] + metric_dict['FN'])
        metric_dict['Precision'] = metric_dict['TP'] / np.maximum(1.0, metric_dict['TP'] + metric_dict['FP'])
        metric_dict['MODA'] = (metric_dict['TP'] - metric_dict['FP']) / np.maximum(1.0,
                                                                                   metric_dict['TP'] + metric_dict['FN'])
        metric_dict['MOTA'] = (metric_dict['TP'] - metric_dict['FP'] - metric_dict['IDSW']
                               ) / np.maximum(1.0, metric_dict['TP'] + metric_dict['FN'])
        metric_dict['MOTP'] = metric_dict['MOTP_sum'] / np.maximum(1.0, metric_dict['TP'])
        metric_dict['sMOTA'] = (metric_dict['MOTP_sum'] - metric_dict['FP'] - metric_dict['IDSW']
                                ) / np.maximum(1.0, metric_dict['TP'] + metric_dict['FN'])

        metric_dict['F1'] = metric_dict['TP'] / np.maximum(1.0,
                                                           metric_dict['TP'] + 0.5*metric_dict['FN'] + 0.5*metric_dict['FP'])
        metric_dict['FP_per_frame'] = metric_dict['FP'] / np.maximum(1.0, metric_dict['FRAMES'])
        safe_log_idsw = np.log10(metric_dict['IDSW']) if metric_dict['IDSW'] > 0 else metric_dict['IDSW']
        metric_dict['MOTAL'] = (metric_dict['TP'] - metric_dict['FP'] - safe_log_idsw) / np.maximum(1.0,
                                                                                                    metric_dict['TP'] + metric_dict['FN'])
        return metric_dict

from typing import Dict, List

import numpy as np
from cv_eval_metrics.config import TMetricConfig
from cv_eval_metrics.base import BaseMetric
from scipy.optimize import linear_sum_assignment


class HOTA(BaseMetric):
    """Class which implements the HOTA metrics"""
    @property
    def metric_fields(self) -> List:
        return self._metric_fields

    def __init__(self) -> None:
        super().__init__()
        self.__array_labels = np.arange(0.05, 0.99, 0.05)
        self.__metric_supp_list = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self._metric_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA']

    def compute(self, cfg: TMetricConfig) -> Dict:
        metric_dict = {}
        for field in self.metric_fields + self.__metric_supp_list:
            metric_dict[field] = np.zeros((len(self.__array_labels)), dtype=np.float)

        # Return result quickly if pred_ids_t or gt sequence is empty
        if cfg.pred_dets_cnt == 0:
            metric_dict['HOTA_FN'] = cfg.gt_dets_cnt * np.ones((len(self.__array_labels)), dtype=np.float)
            metric_dict['LocA'] = np.ones((len(self.__array_labels)), dtype=np.float)
            return metric_dict
        if cfg.gt_dets_cnt == 0:
            metric_dict['HOTA_FP'] = cfg.pred_dets_cnt * np.ones((len(self.__array_labels)), dtype=np.float)
            metric_dict['LocA'] = np.ones((len(self.__array_labels)), dtype=np.float)
            return metric_dict

        # Variables counting global association
        potential_matches_count = np.zeros((cfg.pred_ids_cnt, cfg.gt_ids_cnt))
        gt_id_count = np.zeros((1, cfg.gt_ids_cnt))
        pred_id_count = np.zeros((cfg.pred_ids_cnt, 1))

        # First loop through each timestep and accumulate global track information.
        for timestamp, (gt_ids_t, pred_ids_t) in enumerate(zip(cfg.gt_ids, cfg.pred_ids)):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = cfg.similarity_scores[timestamp]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[pred_ids_t[:, np.newaxis], gt_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and pred_ids_t_id.
            gt_id_count[0, gt_ids_t] += 1
            pred_id_count[pred_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + pred_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.__array_labels]

        # Calculate scores for each timestep
        for timestamp, (gt_ids_t, pred_ids_t) in enumerate(zip(cfg.gt_ids, cfg.pred_ids)):
            # Deal with the case that there are no gt_det/pred_ids_t_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.__array_labels):
                    metric_dict['HOTA_FP'][a] += len(pred_ids_t)
                continue
            if len(pred_ids_t) == 0:
                for a, alpha in enumerate(self.__array_labels):
                    metric_dict['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = cfg.similarity_scores[timestamp]
            score_mat = global_alignment_score[pred_ids_t[:, np.newaxis], gt_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.__array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                metric_dict['HOTA_TP'][a] += num_matches
                metric_dict['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                metric_dict['HOTA_FP'][a] += len(pred_ids_t) - num_matches
                if num_matches > 0:
                    metric_dict['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][pred_ids_t[alpha_match_rows], gt_ids_t[alpha_match_cols]] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/pred_ids_t_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.__array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + pred_id_count - matches_count)
            metric_dict['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, metric_dict['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            metric_dict['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, metric_dict['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, pred_id_count)
            metric_dict['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, metric_dict['HOTA_TP'][a])

        # Calculate final scores
        metric_dict['LocA'] = np.maximum(1e-10, metric_dict['LocA']) / np.maximum(1e-10, metric_dict['HOTA_TP'])
        metric_dict = self._compute_final_fields(metric_dict)

        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        """Combines metrics across all sequences"""
        metric_dict = {}
        for field in self.__metric_supp_list:
            metric_dict[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            metric_dict[field] = self._combine_weighted_avg(all_res, field, metric_dict, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        metric_dict['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, metric_dict['HOTA_TP'])
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    @ staticmethod
    def _compute_final_fields(metric_dict: Dict) -> Dict:
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        metric_dict['DetRe'] = metric_dict['HOTA_TP'] / np.maximum(1, metric_dict['HOTA_TP'] + metric_dict['HOTA_FN'])
        metric_dict['DetPr'] = metric_dict['HOTA_TP'] / np.maximum(1, metric_dict['HOTA_TP'] + metric_dict['HOTA_FP'])
        metric_dict['DetA'] = metric_dict['HOTA_TP'] / np.maximum(
            1, metric_dict['HOTA_TP'] + metric_dict['HOTA_FN'] + metric_dict['HOTA_FP'])
        metric_dict['HOTA'] = np.sqrt(metric_dict['DetA'] * metric_dict['AssA'])
        metric_dict['RHOTA'] = np.sqrt(metric_dict['DetRe'] * metric_dict['AssA'])

        return metric_dict

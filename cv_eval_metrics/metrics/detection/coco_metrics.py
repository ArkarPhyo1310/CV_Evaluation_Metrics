from typing import Any, Dict, Optional, List, Tuple
import numpy as np

from cv_eval_metrics.base import BaseMetric
from cv_eval_metrics.config import DMetricConfig
from cv_eval_metrics.utils.bboxes import box_iou, box_area


class COCOMetrics(BaseMetric):
    @property
    def metric_fields(self) -> List[str]:
        return self._metric_fields

    def __init__(self) -> None:
        super(COCOMetrics, self).__init__()
        self._metric_fields = [
            "mAP", "mAP:0.5", "mAP:0.75", "mAP(small)", "mAP(medium)", "mAP(large)",
            "mAR:1", "mAR:10", "mAR:100", "mAR(small)", "mAR(medium)", "mAR(large)",
        ]
        self.__metric_supp_list = ["precision", "recall"]

    def _compute_iou(self, img_id: int, class_id: int):
        gt = self.__gt_bboxes[img_id]
        pred = self.__pred_bboxes[img_id]

        gt_label = self.__gt_labels[img_id] == class_id
        pred_label = self.__pred_labels[img_id] == class_id

        if len(gt_label) == 0 and len(pred_label) == 0:
            return []

        gt = gt[gt_label]
        pred = pred[pred_label]

        if len(gt) == 0 and len(pred) == 0:
            return []

        scores = self.__pred_scores[img_id]
        scores_filtered = scores[self.__pred_labels[img_id] == class_id]
        scores_inds = np.argsort(-scores_filtered, kind='mergesort')
        pred = pred[scores_inds]

        if len(pred) > self.__last_max_dets:
            pred = pred[:self.__last_max_dets]

        ious = box_iou(pred, gt, box_format=self.__box_format)
        return ious

    def _evaluate_image_gt_no_preds(
        self, gt: np.ndarray, gt_label_mask: np.ndarray, area_range: Tuple[int, int], nb_iou_thrs: int
    ) -> Dict[str, Any]:
        """Some GT but no predictions."""
        # GTs
        gt = gt[gt_label_mask]
        nb_gt = len(gt)

        gt_areas = box_area(gt)
        ignore_area = (gt_areas < area_range[0]) | (gt_areas > area_range[1])
        gt_ignore = np.sort(ignore_area)
        gt_inds = np.argsort(ignore_area, kind='mergesort')

        # Detections
        nb_det = 0
        det_ignore = np.zeros((nb_iou_thrs, nb_det))

        return {
            "predMatches": np.zeros((nb_iou_thrs, nb_det)),
            "gtMatches": np.zeros((nb_iou_thrs, nb_gt)),
            "predScores": np.zeros(nb_det),
            "gtIgnores": gt_ignore,
            "predIgnores": det_ignore,
        }

    def _evaluate_image_preds_no_gt(
        self, det: np.ndarray, idx: int, det_label_mask: np.ndarray, max_det: int, area_range: Tuple[int, int], nb_iou_thrs: int
    ) -> Dict[str, Any]:
        """Some predictions but no GT."""
        # GTs
        nb_gt = 0
        gt_ignore = np.zeros(nb_gt)

        # Detections
        det = det[det_label_mask]
        scores = self.__pred_scores[idx]
        scores_filtered = scores[det_label_mask]
        scores_sorted = np.sort(scores_filtered)[::-1]
        dtind = np.argsort(-scores_filtered, kind='mergesort')

        det = det[dtind]
        if len(det) > max_det:
            det = det[:max_det]
        nb_det = len(det)

        det_areas = box_area(det)
        pred_ignores_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
        ar = np.reshape(pred_ignores_area, (1, nb_det))
        det_ignore = np.repeat(ar, nb_iou_thrs, 0)

        return {
            "predMatches": np.zeros((nb_iou_thrs, nb_det)),
            "gtMatches": np.zeros((nb_iou_thrs, nb_gt)),
            "predScores": scores_sorted,
            "gtIgnores": gt_ignore,
            "predIgnores": det_ignore,
        }

    def _evaluate_img(self, img_id: int, class_id: int, area: list):
        gt = self.__gt_bboxes[img_id]
        pred = self.__pred_bboxes[img_id]

        gt_label = self.__gt_labels[img_id] == class_id
        pred_label = self.__pred_labels[img_id] == class_id

        if len(gt_label) == 0 and len(pred_label) == 0:
            return None

        # Some GT but no predictions
        if len(gt_label) > 0 and len(pred_label) == 0:
            return self._evaluate_image_gt_no_preds(gt, gt_label, area, len(self.__iou_thres))

        # Some predictions but no GT
        if len(gt_label) == 0 and len(pred_label) >= 0:
            return self._evaluate_image_preds_no_gt(
                pred, img_id, gt_label, self.__last_max_dets, area, len(self.__iou_thres))

        gt = gt[gt_label]
        pred = pred[pred_label]

        if len(gt) == 0 and len(pred) == 0:
            return None

        gt_areas = box_area(gt)
        ignore_area = (gt_areas < area[0]) | (gt_areas > area[1])
        ignore_area_sorted = np.sort(ignore_area)
        gt_inds = np.argsort(ignore_area, kind='mergesort')

        gt = gt[gt_inds]
        scores = self.__pred_scores[img_id]
        scores_filtered = scores[pred_label]
        scores_sorted = np.sort(scores_filtered)[::-1]
        pred_inds = np.argsort(-scores_filtered, kind='mergesort')

        pred = pred[pred_inds]

        if len(pred) > self.__last_max_dets:
            pred = pred[:self.__last_max_dets]

        ious = self.__ious[img_id, class_id][:, gt_inds] if len(
            self.__ious[img_id, class_id]) > 0 else self.__ious[img_id, class_id]

        iou_thres_cnt = len(self.__iou_thres)
        gt_cnt = len(gt)
        pred_cnt = len(pred)

        gt_matches = np.zeros((iou_thres_cnt, gt_cnt))
        pred_matches = np.zeros((iou_thres_cnt, pred_cnt))
        gt_ignores = ignore_area_sorted
        pred_ignores = np.zeros((iou_thres_cnt, pred_cnt))

        if not len(ious) == 0:
            for thres_idx, thres in enumerate(self.__iou_thres):
                for pred_idx, _ in enumerate(pred):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([thres, 1-1e-10])
                    m = -1
                    for gt_idx, _ in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gt_matches[thres_idx, gt_idx] > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ignores[m] == 0 and gt_ignores[gt_idx] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[pred_idx, gt_idx] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[pred_idx, gt_idx]
                        m = gt_idx
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue

                    pred_ignores[thres_idx, pred_idx] = gt_ignores[m]
                    pred_matches[thres_idx, pred_idx] = 1
                    gt_matches[thres_idx, m] = 1

        pred_areas = box_area(pred)
        pred_ignores_area = (pred_areas < area[0]) | (pred_areas > area[1])
        ar = np.reshape(pred_ignores_area, (1, pred_cnt))
        pred_ignores = np.logical_or(pred_ignores, np.logical_and(pred_matches == 0, np.repeat(ar, iou_thres_cnt, 0)))

        return {
            'predMatches': pred_matches,
            'predScores': scores_sorted,
            'predIgnores': pred_ignores,
            'gtMatches': gt_matches,
            'gtIgnores': gt_ignores
        }

    def _calculate_recall_prescision_scores(
        self, cls_idx: int, area_idx: int, max_det_thres_idx: int, area_ranges_cnt: int, imgs_cnt: int, max_det: int
    ):
        rec_thres_cnt = len(self.__rec_thres)
        idx_cls_ptr = cls_idx * area_ranges_cnt * imgs_cnt
        idx_area_ptr = area_idx * imgs_cnt

        img_eval_cls_bbox = [
            self.__eval_imgs[idx_cls_ptr + idx_area_ptr + i] for i in range(imgs_cnt)
        ]
        img_eval_cls_bbox = [e for e in img_eval_cls_bbox if e is not None]
        if not img_eval_cls_bbox:
            return
        pred_scores = np.concatenate([e['predScores'][:max_det] for e in img_eval_cls_bbox])
        pred_inds = np.argsort(-pred_scores, kind='mergesort')
        pred_scores_sorted = pred_scores[pred_inds]

        pred_matches = np.concatenate([e['predMatches'][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, pred_inds]
        pred_ignores = np.concatenate([e["predIgnores"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, pred_inds]
        gt_ignores = np.concatenate([e["gtIgnores"] for e in img_eval_cls_bbox])
        npig = np.count_nonzero(gt_ignores == False)
        if npig == 0:
            return
        true_positives = np.logical_and(pred_matches, np.logical_not(pred_ignores))
        false_positives = np.logical_and(np.logical_not(pred_matches), np.logical_not(pred_ignores))

        tp_sum = np.cumsum(true_positives, axis=1, dtype=float)
        fp_sum = np.cumsum(false_positives, axis=1, dtype=float)

        for idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            nd = len(tp)
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))

            prec = np.zeros((rec_thres_cnt, ))  # q
            score = np.zeros((rec_thres_cnt, ))  # ss

            self.__recall[idx, cls_idx, area_idx, max_det_thres_idx] = rc[-1] if nd else 0

            pr = pr.tolist()
            prec = prec.tolist()

            for i in range(nd - 1, 0, -1):
                if pr[i] > pr[i-1]:
                    pr[i-1] = pr[i]
            inds = np.searchsorted(rc, self.__rec_thres, side="left")
            try:
                for ri, pi in enumerate(inds):
                    prec[ri] = pr[pi]
                    score[ri] = pred_scores_sorted[pi]
            except:
                pass

            self.__precision[idx, :, cls_idx, area_idx, max_det_thres_idx] = np.array(prec)
            self.__scores[idx, :, cls_idx, area_idx, max_det_thres_idx] = np.array(score)

    def _summarize(
        self,
        metric_dict: dict,
        ap: bool = True,
        iou_threshold: Optional[float] = None,
        area_ranges: str = "all",
        max_dets: int = 100
    ) -> np.ndarray:
        area_inds = [i for i, k in enumerate(self.__area_ranges.keys()) if k == area_ranges]
        mdet_inds = [i for i, k in enumerate(self.__max_dets) if k == max_dets]

        if ap:
            res = metric_dict["precision"]
            if iou_threshold is not None:
                thres = np.where(iou_threshold == self.__iou_thres)[0]
                res = res[thres]
            res = res[:, :, :, area_inds, mdet_inds]
        else:
            res = metric_dict["recall"]
            if iou_threshold is not None:
                thres = np.where(iou_threshold == self.__iou_thres)[0]
                res = res[thres]
            res = res[:, :, area_inds, mdet_inds]

        if len(res[res > -1]) == 0:
            mean_res = -1
        else:
            mean_res = np.mean(res[res > -1])

        return mean_res

    def compute(self, cfg: DMetricConfig) -> Dict:
        self.__max_dets = cfg.max_dets
        self.__last_max_dets = cfg.max_dets[-1]
        self.__iou_thres = cfg.iou_thresholds
        self.__rec_thres = cfg.rec_thresholds
        self.__gt_bboxes = cfg.gt_bboxes
        self.__gt_labels = cfg.gt_labels

        self.__pred_bboxes = cfg.pred_bboxes
        self.__pred_labels = cfg.pred_labels
        self.__pred_scores = cfg.pred_scores

        self.__area_ranges = cfg.area_scales
        self.__box_format = cfg.bbox_format

        img_ids = range(len(self.__gt_bboxes))
        area_ranges = cfg.area_scales.values()

        metric_dict = {}

        for field in self._metric_fields + self.__metric_supp_list:
            metric_dict[field] = -1

        self.__ious = {
            (img_id, class_id): self._compute_iou(img_id, class_id)
            for img_id in img_ids
            for class_id in cfg.classes
        }

        self.__eval_imgs = [
            self._evaluate_img(img_id, class_id, area)
            for class_id in cfg.classes
            for area in area_ranges
            for img_id in img_ids
        ]

        iou_thres_cnt = len(self.__iou_thres)
        rec_thres_cnt = len(self.__rec_thres)
        classes_cnt = len(cfg.classes)
        area_ranges_cnt = len(area_ranges)
        max_dets_cnt = len(cfg.max_dets)
        imgs_cnt = len(img_ids)

        self.__precision = -np.ones((iou_thres_cnt, rec_thres_cnt, classes_cnt, area_ranges_cnt, max_dets_cnt))
        self.__recall = -np.ones((iou_thres_cnt, classes_cnt, area_ranges_cnt, max_dets_cnt))
        self.__scores = -np.ones((iou_thres_cnt, rec_thres_cnt, classes_cnt, area_ranges_cnt, max_dets_cnt))

        for cls_idx, _ in enumerate(cfg.classes):
            for area_idx, _ in enumerate(cfg.area_scales):
                for max_det_thres_idx, max_det in enumerate(cfg.max_dets):
                    self._calculate_recall_prescision_scores(
                        cls_idx, area_idx, max_det_thres_idx, area_ranges_cnt, imgs_cnt, max_det
                    )

        metric_dict["precision"] = self.__precision
        metric_dict["recall"] = self.__recall

        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    def combine_result(self, all_res: Dict) -> Dict:
        """Combines metrics across all sequences"""
        metric_dict = {}
        for field in self.__metric_supp_list:
            metric_dict[field] = self._combine_sum(all_res, field)
        metric_dict = self._compute_final_fields(metric_dict)
        return metric_dict

    def _compute_final_fields(self, metric_dict):
        metric_dict["mAP"] = self._summarize(metric_dict, True)
        if 0.5 in self.__iou_thres:
            metric_dict["mAP:0.5"] = self._summarize(metric_dict, True, 0.5, max_dets=self.__last_max_dets)
        if 0.75 in self.__iou_thres:
            metric_dict["mAP:0.75"] = self._summarize(metric_dict, True, 0.75, max_dets=self.__last_max_dets)

        for max_det in self.__max_dets:
            metric_dict[f"mAR:{max_det}"] = self._summarize(metric_dict, False, max_dets=max_det)

        for area in self.__area_ranges.keys():
            metric_dict[f"mAP({area})"] = self._summarize(
                metric_dict, True, area_ranges=f"{area}", max_dets=self.__last_max_dets)
            metric_dict[f"mAR({area})"] = self._summarize(
                metric_dict, False, area_ranges=f"{area}", max_dets=self.__last_max_dets
            )

        return metric_dict

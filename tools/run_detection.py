""" run_detection.py

Run example:
python run_detection.py

"""
import numpy as np
from cv_eval_metrics.base.evaluator import MetricEvaluator
from cv_eval_metrics.config import DMetricConfig
from cv_eval_metrics.dataset import COCO
from cv_eval_metrics.objects import DetectionObject

if __name__ == "__main__":

    gt_path = "./data/detection/instances_val2014_modified.json"
    pred_path = "./data/detection/instances_val2014_fakebbox100_results.json"
    coco_dataset = COCO(gt_path, pred_path, 'json')
    det_cfg = DMetricConfig(classes=[i for i in range(1, 81)], bbox_format="xywh")
    evaluator = MetricEvaluator(evaluation_task='detection', benchmark='coco')

    # pred = DetectionObject(bboxes=[[258.0, 41.0, 606.0, 285.0]], labels=[0],
    #                        scores=[0.536])
    # gt = DetectionObject(
    #     bboxes=[[214.0, 41.0, 562.0, 285.0]],
    #     labels=[0])

    gt, pred = coco_dataset.process()
    det_cfg.update(gt, pred)
    evaluator.evaluate(det_cfg, curr_seq="val2014")
    evaluator.render_result(model_name="Custom", show_overall=False)

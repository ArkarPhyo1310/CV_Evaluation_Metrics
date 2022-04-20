
""" run_tracking.py

Run example:
python run_tracking.py
"""
from cv_eval_metrics.abstract import MetricEvaluator
from cv_eval_metrics.config import MetricEvalConfig, TMetricConfig
from cv_eval_metrics.dataset import MOT

if __name__ == '__main__':
    gt_path = "./data/gt/MOT17-train/"
    pred_path = "./data/trackers/MPNTrack/"

    metric_cfg = TMetricConfig(threshold=0.5)

    mot_dataset = MOT(gt_path=gt_path, pred_path=pred_path, file_format="txt")

    eval_cfg = MetricEvalConfig(benchmark="MOT", evaluation_task="tracking")
    evaluator = MetricEvaluator(eval_cfg)

    gt_files, pred_files, seq_names = mot_dataset.gt_files, mot_dataset.pred_files, mot_dataset.seq_list

    for gt_file, pred_file, seq in zip(gt_files, pred_files, seq_names):
        mot_dataset.process(gt_file, pred_file)
        mot_dataset.assign(metric_cfg)
        evaluator.evaluate(metric_cfg, curr_seq=seq)

    evaluator.render_result(model_name="MPNTrack")

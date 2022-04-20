""" run_classification.py

Run example:
python run_classification.py

"""
import numpy as np
from cv_eval_metrics.abstract.evaluator import MetricEvaluator
from cv_eval_metrics.config import CMetricConfig, MetricEvalConfig
from cv_eval_metrics.objects import ClassificationObject

if __name__ == "__main__":
    # target = [1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2, 0, 0]
    # pred = [1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1, 0, 5]

    # target = ['cat', 'cat', 'dog', 'pig', 'pig']
    # pred = ['cat', 'dog', 'dog', 'cat', 'pig']

    pred = np.array([[0.75, 0.15, 0.10, 0.0],
                    [0.15, 0.75, 0.0, 0.10],
                    [0.15, 0.10, 0.75, 0.0],
                    [0.0, 0.10, 0.15, 0.75]])
    target = [0, 1, 3, 2]

    gt = ClassificationObject(labels=target)
    pred = ClassificationObject(labels=pred)

    cls_cfg = CMetricConfig(
        classes=[i for i in range(4)], top_k=2, average="macro"
    )

    cls_cfg.update(gt, pred)

    metric_list = ["Accuracy", "Precision", "Recall", "F1-score", "Confusion Matrix"]
    eval_cfg = MetricEvalConfig(specific_metric_fields=metric_list, evaluation_task="classification")
    evaluator = MetricEvaluator(eval_cfg=eval_cfg)
    evaluator.evaluate(cls_cfg)
    evaluator.render_result(model_name="Custom", show_overall=False)

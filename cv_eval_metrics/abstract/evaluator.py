from typing import Union

import numpy as np
import pandas as pd
from cv_eval_metrics.config import MetricEvalConfig, TMetricConfig
from cv_eval_metrics.config.c_metric_cfg import CMetricConfig
from cv_eval_metrics.metrics.classification import (Accuracy,
                                                          ConfusionMatrix,
                                                          F1Score, Precision,
                                                          Recall)
from cv_eval_metrics.metrics.common import COUNT
from cv_eval_metrics.metrics.tracking import CLEAR, HOTA, IDENTITY
from tabulate import tabulate


class MetricEvaluator:
    """Evaluator class for evaluating different metrics"""

    def __init__(self, eval_cfg: MetricEvalConfig) -> None:
        self.evaluation_result = {}
        self.table_result = {}
        self.tracking_benchmark = eval_cfg.benchmark
        self._check_config(eval_cfg)

    def evaluate(self, metric_cfg: Union[TMetricConfig, CMetricConfig], curr_seq: str = None):
        self.classes = metric_cfg.classes
        if curr_seq is None:
            curr_seq = 'N/A'

        self.evaluation_result[curr_seq] = {}

        for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
            self.evaluation_result[curr_seq][metric_name] = metric.compute(metric_cfg)

    def render_result(self, model_name: str, show_overall: bool = True):
        self._summarize_result(model_name, show_overall)
        if len(self.specific_metric_fields) != 0:
            self.show_specific_fields(model_name)
        else:
            for metric_name in self.table_result.keys():
                if metric_name == "ConfusionMatrix":
                    self.show_cm(self.table_result[metric_name])
                else:
                    print(tabulate(self.table_result[metric_name]['data'],
                                   headers=self.table_result[metric_name]['header'],
                                   tablefmt="pretty"), end="\n")

    def show_cm(self, result: dict):
        headers = ["Confusion Matrix"] + self.classes
        data = eval(result['data'].squeeze()[-1])

        print(tabulate(data, headers=headers, showindex=self.classes, tablefmt='pretty'))

    def show_specific_fields(self, model_name: str = None):
        data_array = self.table_result['COUNT']['data'][:, 1:]
        index = self.table_result['COUNT']['data'][:, 0]
        headers = self.table_result['COUNT']['header'][1:]

        for metric_name in self.table_result.keys():
            if metric_name == "COUNT":
                continue
            if metric_name == "ConfusionMatrix":
                self.show_cm(self.table_result[metric_name])
            else:
                np_ta = self.table_result[metric_name]['data'][:, 1:]
                data_array = np.hstack([data_array, np_ta])
                headers = np.hstack([headers, self.table_result[metric_name]['header'][1:]])

        df = pd.DataFrame(data_array, index=index, columns=headers)

        df = df[self.specific_metric_fields]
        if self.tracking_benchmark is None:
            self.specific_metric_fields = ["MODEL NAME: " + model_name] + self.specific_metric_fields
        else:
            self.specific_metric_fields = [self.tracking_benchmark + f": {model_name}"] + self.specific_metric_fields
        print(tabulate(df, headers=self.specific_metric_fields, tablefmt="pretty"))

    def _summarize_result(self, model_name: str, show_overall: bool = True):
        self.evaluation_result['COMBINED_SEQ'] = {}

        # combine sequences
        self.evaluation_result['COMBINED_SEQ'] = {}
        for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
            current_res = {seq_key: seq_value[metric_name] for seq_key,
                           seq_value in self.evaluation_result.items() if seq_key != 'COMBINED_SEQ'}
            self.evaluation_result['COMBINED_SEQ'][metric_name] = metric.combine_result(current_res)

        num_preds = self.evaluation_result['COMBINED_SEQ']['COUNT']['Pred_Cnts']
        if num_preds > 0:
            for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
                table_res = {seq_key: seq_value[metric_name] for seq_key, seq_value
                             in self.evaluation_result.items()}

                data, header = metric.record_table(table_res, model_name, show_overall)
                self.table_result[metric_name] = {
                    'data': np.array(data),
                    'header': np.array(header)
                }

    def _check_config(self, eval_cfg: MetricEvalConfig):
        metric_cls_set: list = list()
        print("Checking Benchmark..Adding Metric Class if required...")
        if self.tracking_benchmark == "MOT":
            eval_cfg.specific_metric_fields = eval_cfg.mot_benchmark_fields
        elif self.tracking_benchmark == "KITTI":
            eval_cfg.specific_metric_fields = eval_cfg.kitti_benchmark_fields
        else:
            print("No benchmark is provided!")

        print("Checking Metric Classes...")
        if len(eval_cfg.metrics_cls_to_eval) != 0:
            metric_cls_set = set(eval_cfg.metrics_cls_to_eval)
        else:
            if eval_cfg.evaluation_task == "tracking":
                print("Checking Specific Metric Fields List...")
                clear_metric = CLEAR()
                hota_metric = HOTA()
                id_metric = IDENTITY()
                if len(eval_cfg.specific_metric_fields) != 0:
                    for field in eval_cfg.specific_metric_fields:
                        if field in clear_metric.metric_fields:
                            metric_cls_set.append(clear_metric)
                        elif field in hota_metric.metric_fields:
                            metric_cls_set.append(hota_metric)
                        elif field in id_metric.metric_fields:
                            metric_cls_set.append(id_metric)
                        else:
                            print(f"There is no '{field}' metric field in current Metric Classes. Skipping...")
                            eval_cfg.specific_metric_fields.remove(field)
            elif eval_cfg.evaluation_task == "classification":
                print("Checking Specific Metric Fields List...")
                acc = Accuracy()
                pre = Precision()
                rcl = Recall()
                f1 = F1Score()
                cm = ConfusionMatrix()
                if len(eval_cfg.specific_metric_fields) != 0:
                    eval_cfg.specific_metric_fields = [field.title() for field in eval_cfg.specific_metric_fields]
                    for field in eval_cfg.specific_metric_fields:
                        if field in acc.metric_fields:
                            metric_cls_set.append(acc)
                        elif field in pre.metric_fields:
                            metric_cls_set.append(pre)
                        elif field in rcl.metric_fields:
                            metric_cls_set.append(rcl)
                        elif field in f1.metric_fields:
                            metric_cls_set.append(f1)
                        elif field in cm.metric_fields:
                            metric_cls_set.append(cm)
                        else:
                            print(f"There is no '{field}' metric field in current Metric Classes. Skipping...")
                            eval_cfg.specific_metric_fields.remove(field)

        self.metric_cls_list = list(set(metric_cls_set)) + [COUNT(task=eval_cfg.evaluation_task)]

        self.metric_names = [metric.get_name() for metric in self.metric_cls_list]
        if "Confusion Matrix".title() in eval_cfg.specific_metric_fields:
            eval_cfg.specific_metric_fields.remove("confusion matrix".title())
        self.specific_metric_fields = eval_cfg.specific_metric_fields

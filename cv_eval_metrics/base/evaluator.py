from typing import List, Optional, Union

import numpy as np
import pandas as pd
from cv_eval_metrics.config import TMetricConfig
from cv_eval_metrics.config.c_metric_cfg import CMetricConfig
from cv_eval_metrics.config.d_metric_cfg import DMetricConfig
from cv_eval_metrics.metrics.classification import (Accuracy,
                                                    ConfusionMatrix,
                                                    F1Score, Precision,
                                                    Recall)
from cv_eval_metrics.metrics.detection import COCOMetrics
from cv_eval_metrics.metrics.common import COUNT
from cv_eval_metrics.metrics.tracking import CLEAR, HOTA, IDENTITY
from tabulate import tabulate


class MetricEvaluator:
    """Evaluator class for evaluating different metrics"""

    def __init__(
        self,
        evaluation_task: str,
        benchmark: Optional[str] = None,
        metric_classes: Optional[List] = None,
        specific_metric_fields: Optional[List[str]] = None,
    ) -> None:
        self.evaluation_result = {}
        self.table_result = {}
        self.benchmark = benchmark.lower() if benchmark is not None else benchmark
        self.evaluation_task = evaluation_task
        self.metrics_cls_to_eval = metric_classes
        self.specific_metric_fields = specific_metric_fields

        self.benchmark_dict = {
            'mot': [
                "MOTA", "IDF1", "HOTA", "MT", "ML", "FP", "FN", "Recall", "Precision",
                "AssA", "DetA", "AssRe", "AssPr", "DetRe", "DetPr", "LocA", "IDSW", "FRAG"
            ],
            'kitti': ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "MOTA"],
            'coco': COCOMetrics().metric_fields
        }

        self._check_config()

    def evaluate(self, metric_cfg: Union[TMetricConfig, CMetricConfig, DMetricConfig], curr_seq: str = None):
        self.classes = metric_cfg.classes
        if curr_seq is None:
            curr_seq = 'N/A'

        self.evaluation_result[curr_seq] = {}
        print(f"Calculating Metric...{self.metric_names} for {curr_seq}")

        for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
            self.evaluation_result[curr_seq][metric_name] = metric.compute(metric_cfg)

    def render_result(self, model_name: str, show_overall: bool = True):
        self._summarize_result(model_name, show_overall)
        if self.specific_metric_fields:
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
        if self.benchmark is None:
            self.specific_metric_fields = ["MODEL NAME: " + model_name] + self.specific_metric_fields
        else:
            self.specific_metric_fields = [self.benchmark + f": {model_name}"] + self.specific_metric_fields
        print(tabulate(df, headers=self.specific_metric_fields, tablefmt="pretty"))

    def _summarize_result(self, model_name: str, show_overall: bool = True):
        self.evaluation_result['COMBINED_SEQ'] = {}

        # combine sequences
        self.evaluation_result['COMBINED_SEQ'] = {}
        for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
            current_res = {seq_key: seq_value[metric_name] for seq_key,
                           seq_value in self.evaluation_result.items() if seq_key != 'COMBINED_SEQ'}
            self.evaluation_result['COMBINED_SEQ'][metric_name] = metric.combine_result(current_res)

        if 'Pred_Cnts' in self.evaluation_result['COMBINED_SEQ']['COUNT']:
            num_preds = self.evaluation_result['COMBINED_SEQ']['COUNT']['Pred_Cnts']
        else:
            num_preds = self.evaluation_result['COMBINED_SEQ']['COUNT']['Pred_Dets']
        if num_preds > 0:
            for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
                table_res = {seq_key: seq_value[metric_name] for seq_key, seq_value
                             in self.evaluation_result.items()}

                data, header = metric.record_table(table_res, model_name, show_overall)
                self.table_result[metric_name] = {
                    'data': np.array(data),
                    'header': np.array(header)
                }

    def _check_config(self):
        metric_cls_set: list = list()
        print("Checking Benchmark..Adding Metric Class if required...")
        if self.benchmark is not None:
            try:
                self.specific_metric_fields = self.benchmark_dict[self.benchmark]
            except Exception:
                raise ValueError(f"'{self.benchmark}' does not implemented yet!")
        else:
            print("No benchmark is provided!")

        print("Checking Metric Classes...")
        if self.metrics_cls_to_eval:
            metric_cls_set = set(self.metrics_cls_to_eval)
        else:
            if self.evaluation_task == "tracking":
                print("Checking Specific Metric Fields List for Object Tracking...")
                clear_metric = CLEAR()
                hota_metric = HOTA()
                id_metric = IDENTITY()
                if self.specific_metric_fields:
                    for field in self.specific_metric_fields:
                        if field in clear_metric.metric_fields:
                            metric_cls_set.append(clear_metric)
                        elif field in hota_metric.metric_fields:
                            metric_cls_set.append(hota_metric)
                        elif field in id_metric.metric_fields:
                            metric_cls_set.append(id_metric)
                        else:
                            print(f"There is no '{field}' metric field in current Metric Classes. Skipping...")
                            self.specific_metric_fields.remove(field)
            elif self.evaluation_task == "classification":
                print("Checking Specific Metric Fields List for Image Classification...")
                acc = Accuracy()
                pre = Precision()
                rcl = Recall()
                f1 = F1Score()
                cm = ConfusionMatrix()
                if self.specific_metric_fields:
                    self.specific_metric_fields = [field.title() for field in self.specific_metric_fields]
                    for field in self.specific_metric_fields:
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
                            self.specific_metric_fields.remove(field)
            elif self.evaluation_task == "detection":
                print("Checking Specific Metric Fields List for Object Detection...")
                coco = COCOMetrics()
                if self.specific_metric_fields:
                    for field in self.specific_metric_fields:
                        if field in coco.metric_fields:
                            metric_cls_set.append(coco)

        self.metric_cls_list = list(set(metric_cls_set)) + [COUNT(task=self.evaluation_task)]

        self.metric_names = [metric.get_name() for metric in self.metric_cls_list]
        if "Confusion Matrix".title() in self.specific_metric_fields:
            self.specific_metric_fields.remove("confusion matrix".title())

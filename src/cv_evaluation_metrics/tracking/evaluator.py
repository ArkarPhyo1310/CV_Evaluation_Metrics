from typing import List

import numpy as np

from cv_evaluation_metrics.config import DatasetCfg
from cv_evaluation_metrics.config.t_metric_cfg import TMetricConfig
from cv_evaluation_metrics.tracking.metrics import COUNT, CLEAR, HOTA, IDENTITY

from tabulate import tabulate
import pandas as pd


class TMetricEvaluator:
    """Evaluator class for evaluating different metrics"""

    def __init__(self, metric_cfg: TMetricConfig) -> None:
        self.evaluation_result = {}
        self.table_result = {}
        self.metric_cfg = metric_cfg
        self.tracking_benchmark = metric_cfg.benchmark
        self._check_config()

    def evaluate(self, dataset_cfg: DatasetCfg, metric_cfg: TMetricConfig):
        self.dataset_name = dataset_cfg.name
        self.tracker_name = dataset_cfg.tracker_name
        self.dataset_class_list = dataset_cfg.classes
        self.current_seq = dataset_cfg.current_seq
        self.current_cls = dataset_cfg.current_class

        # output_folder = dataset_cfg.output_folder
        print(metric_cfg.pred_ids)
        exit()

        self.output_result = {}
        self.output_result[self.dataset_name] = {}
        self.evaluation_result[self.current_seq] = {}
        self.evaluation_result[self.current_seq][self.current_cls] = {}

        for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
            self.evaluation_result[self.current_seq][self.current_cls][metric_name] = metric.run_eval(metric_cfg)

    def summarize_result(self):
        # Combine results over all sequences and then over all classes
        combined_cls_keys: List = []
        self.evaluation_result['COMBINED_SEQ'] = {}
        # combine sequences for each class
        for c_cls in self.dataset_class_list:
            self.evaluation_result['COMBINED_SEQ'][c_cls] = {}
            for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
                current_res = {seq_key: seq_value[c_cls][metric_name] for seq_key,
                               seq_value in self.evaluation_result.items() if seq_key != 'COMBINED_SEQ'}
                self.evaluation_result['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequence(current_res)

        for c_cls in self.evaluation_result['COMBINED_SEQ'].keys():
            num_dets = self.evaluation_result['COMBINED_SEQ'][c_cls]['COUNT']['Pred_Dets']
            if num_dets > 0:
                for metric, metric_name in zip(self.metric_cls_list, self.metric_names):
                    # for combined classes there is no per sequence evaluation
                    if c_cls in combined_cls_keys:
                        table_res = {'COMBINED_SEQ': self.evaluation_result['COMBINED_SEQ'][c_cls][metric_name]}
                    else:
                        table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value
                                     in self.evaluation_result.items()}

                    data, header = metric.record_table(table_res, self.tracker_name, c_cls)
                    self.table_result[metric_name] = {
                        'data': np.array(data),
                        'header': np.array(header)
                    }

    def render_result(self):
        if len(self.metric_cfg.metric_cls_to_eval) != 0:
            for metric_name in self.table_result.keys():
                print(tabulate(self.table_result[metric_name]['data'],
                      headers=self.table_result[metric_name]['header']), end="\n")
        if len(self.specific_metric_fields) != 0:
            self.show_specific_fields()

    def show_specific_fields(self):
        data_array = self.table_result['COUNT']['data'][:, 1:]
        index = self.table_result['COUNT']['data'][:, 0]
        headers = self.table_result['COUNT']['header'][1:]

        for metric_name in self.table_result.keys():
            if metric_name == "COUNT":
                continue
            np_ta = self.table_result[metric_name]['data'][:, 1:]
            data_array = np.hstack([data_array, np_ta])
            headers = np.hstack([headers, self.table_result[metric_name]['header'][1:]])

        df = pd.DataFrame(data_array, index=index, columns=headers)
        df = df[self.specific_metric_fields]
        self.specific_metric_fields.insert(0, self.tracking_benchmark)
        print(tabulate(df, headers=self.specific_metric_fields))

    def _check_config(self):
        metric_cls_set: list = list()
        print("Checking Benchmark..Adding Metric Class if required...")
        if self.tracking_benchmark == "MOT":
            self.metric_cfg.specific_metric_fields = self.metric_cfg.mot_benchmark_fields
        elif self.tracking_benchmark == "KITTI":
            self.metric_cfg.specific_metric_fields = self.metric_cfg.kitti_benchmark_fields
        else:
            print("No benchmark is provided!")

        print("Checking Metric Classes...")
        if len(self.metric_cfg.metric_cls_to_eval) != 0:
            metric_cls_set = set(self.metric_cfg.metric_cls_to_eval)
        else:
            print("Checking Specific Metric Fields List...")
            clear_metric = CLEAR()
            hota_metric = HOTA()
            id_metric = IDENTITY()
            if len(self.metric_cfg.specific_metric_fields) != 0:
                for field in self.metric_cfg.specific_metric_fields:
                    if field in clear_metric.metric_list:
                        metric_cls_set.append(clear_metric)
                    elif field in hota_metric.metric_list:
                        metric_cls_set.append(hota_metric)
                    elif field in id_metric.metric_list:
                        metric_cls_set.append(id_metric)
                    else:
                        print(f"There is no '{field}' metric field in current Metric Classes. Skipping...")
                        self.metric_cfg.specific_metric_fields.remove(field)
            else:
                pass

        self.metric_cls_list = list(set(metric_cls_set)) + [COUNT()]
        self.metric_names = [metric.get_name() for metric in self.metric_cls_list]
        self.specific_metric_fields = self.metric_cfg.specific_metric_fields

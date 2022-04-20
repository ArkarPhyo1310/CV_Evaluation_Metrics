from typing import List


class MetricEvalConfig:
    @property
    def mot_benchmark_fields(self) -> List[str]:
        return self.__mot_benchmark_fields

    @property
    def kitti_benchmark_fields(self) -> List[str]:
        return self.__kitti_benchmark_fields

    @property
    def benchmark(self) -> str:
        return self.__benchmark

    @property
    def metrics_cls_to_eval(self) -> List:
        return self.__metrics_cls_to_eval

    @property
    def evaluation_task(self) -> str:
        return self.__evaluation_task

    @property
    def specific_metric_fields(self) -> List[str]:
        return self.__specific_metric_fields

    @specific_metric_fields.setter
    def specific_metric_fields(self, values: List[str]) -> List[str]:
        self.__specific_metric_fields = values

    def __init__(self,
                 evaluation_task: str,
                 benchmark: str = None,
                 metric_classes: List = [],
                 specific_metric_fields: List[str] = [],
                 ) -> None:
        self.__evaluation_task = evaluation_task
        self.__benchmark: str = benchmark
        self.__metrics_cls_to_eval: List = metric_classes
        self.__specific_metric_fields: List[str] = specific_metric_fields

        self.__kitti_benchmark_fields: List[str] = [
            "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "MOTA"
        ]
        self.__mot_benchmark_fields: List[str] = [
            "MOTA", "IDF1", "HOTA", "MT", "ML", "FP", "FN", "Recall", "Precision",
            "AssA", "DetA", "AssRe", "AssPr", "DetRe", "DetPr", "LocA", "IDSW", "FRAG"
        ]

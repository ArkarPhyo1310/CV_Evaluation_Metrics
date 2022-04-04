from typing import Dict, List


class TMetricConfig:
    @property
    def mot_benchmark_fields(self) -> List[str]:
        return self.__mot_benchmark_fields

    @property
    def kitti_benchmark_fields(self) -> List[str]:
        return self.__kitti_benchmark_fields

    def __init__(self) -> None:
        self.pred_ids: List = []
        self.pred_dets: List = []
        self.pred_ids_cnt: int = 0
        self.pred_dets_cnt: int = 0

        self.gt_dets: List = []
        self.gt_ids: List = []
        self.gt_ids_cnt: int = 0
        self.gt_dets_cnt: int = 0

        self.timestamps_cnt: int = 0
        self.similarity_scores: List = []

        self.threshold: float = 0.5
        self.use_parallel: bool = False
        self.num_parallel_cores: int = 8
        self.benchmark: str = None
        self.metric_cls_to_eval: List = []

        self.specific_metric_fields: List = []

        self.__kitti_benchmark_fields: List[str] = [
            "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "MOTA"
        ]
        self.__mot_benchmark_fields: List[str] = [
            "MOTA", "IDF1", "HOTA", "MT", "ML", "FP", "FN", "Recall", "Precision",
            "AssA", "DetA", "AssRe", "AssPr", "DetRe", "DetPr", "LocA", "IDSW", "FRAG"
        ]
        


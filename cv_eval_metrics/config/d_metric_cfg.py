from typing import List, Optional

import numpy as np


class DMetricConfig:
    def __init__(
        self,
        bbox_format: str = "xyxy",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        class_metrics: bool = False
    ) -> None:
        self.__bbox_format = bbox_format
        self.__iou_thresholds = iou_thresholds or np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.__rec_thresholds = rec_thresholds or np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.1)) + 1, endpoint=True
        )

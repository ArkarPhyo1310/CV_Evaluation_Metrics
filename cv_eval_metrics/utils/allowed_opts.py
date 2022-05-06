from typing import List


bbox_format: List[str] = ["xyxy", "xywh", "cxcywh"]
cm_normalize: List[str] = ["pred", "true", "all"]
weight_average: List[str] = ["macro", "micro"]
mdmc_average: List[str] = ["global", "samplewise"]

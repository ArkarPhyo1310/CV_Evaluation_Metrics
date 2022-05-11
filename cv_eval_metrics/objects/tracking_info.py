from typing import List

import numpy as np


class TrackingObject:
    @property
    def dets(self) -> List[np.ndarray]:
        return self.__dets

    @property
    def ids(self) -> List[np.ndarray]:
        return self.__ids

    @property
    def num_dets(self) -> int:
        return self.__num_dets

    @property
    def num_ids(self) -> int:
        return self.__num_ids

    @property
    def scores(self) -> List[float]:
        return self.__scores

    def __init__(
            self,
            dets: List[np.ndarray],
            ids: List[np.ndarray],
            num_dets: int, num_ids: int,
            scores: List[float] = None) -> None:
        self.__dets = dets
        self.__ids = ids
        self.__num_dets = num_dets
        self.__num_ids = num_ids
        self.__scores = scores

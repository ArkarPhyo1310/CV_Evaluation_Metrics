from abc import ABC
from typing import List

import numpy as np


class ClassificationObject(ABC):
    @property
    def file_names(self) -> List[str]:
        return self.__file_names

    @property
    def labels(self) -> List[str]:
        return self.__labels

    def __init__(
        self,
        labels: List[int] = None,
        file_names: List[str] = None,
    ) -> None:
        self.__file_names = file_names
        self.__labels = labels

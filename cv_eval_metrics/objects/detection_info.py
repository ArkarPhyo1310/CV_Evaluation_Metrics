class DetectionObject:
    @property
    def bboxes(self) -> list:
        return self.__bboxes

    @property
    def labels(self) -> list:
        return self.__labels

    @property
    def scores(self) -> list:
        return self.__scores

    def __init__(
        self,
        bboxes: list = None,
        labels: list = None,
        scores: list = None,
    ) -> None:
        self.__bboxes = bboxes
        self.__labels = labels
        self.__scores = scores

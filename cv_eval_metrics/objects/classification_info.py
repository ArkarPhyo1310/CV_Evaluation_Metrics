class ClassificationObject:
    @property
    def file_names(self) -> list:
        return self.__file_names

    @property
    def labels(self) -> list:
        return self.__labels

    def __init__(
        self,
        labels: list = None,
        file_names: list = None,
    ) -> None:
        self.__file_names = file_names
        self.__labels = labels

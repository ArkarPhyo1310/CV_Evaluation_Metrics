from typing import List


class DatasetCfg:
    def __init__(self) -> None:
        self.name: str = None
        self.classes: List[str] = ['pedestrian']
        self.sequences: List = []
        self.output_folder: str = './output'
        self.current_seq: str = None
        self.current_class: str = None

        self.tracker_name: str = None
        self.detector_name: str = None
        self.segmentor_name: str = None
        self.pose_detector_name: str = None
        self.image_denoiser_name: str = None
        self.anomaly_detector_name: str = None

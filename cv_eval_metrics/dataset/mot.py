import os
from typing import List

import numpy as np
import pandas as pd

from cv_eval_metrics.base import BaseDataset
from cv_eval_metrics.objects.tracking_info import TrackingObject


class MOT(BaseDataset):
    @property
    def gt_files(self) -> List[str]:
        return self.__gt_files

    @property
    def pred_files(self) -> List[str]:
        return self.__pred_files

    @property
    def seq_list(self) -> List[str]:
        return self.__seq_list

    def __init__(self, gt_path: str, pred_path: str, file_format: str) -> None:
        super().__init__(gt_path, pred_path, file_format)
        self._gt_path = gt_path
        self._pred_path = pred_path
        self._file_format = file_format
        self.__valid_labels = [1]

        self.__gt_files, self.__pred_files = self._check_path()
        self.__seq_list = [os.path.splitext(os.path.basename(gt_file))[0] for gt_file in self.__gt_files]

    def process(self, gt_file: str, pred_file: str):
        gt_data: TrackingObject = self.read_mot_file(gt_file, is_gt=True)
        pred_data: TrackingObject = self.read_mot_file(pred_file, is_gt=False)

        return gt_data, pred_data

    def read_mot_file(self, file: str,  is_gt: bool):
        results_dict = dict(
            dets=[],
            ids=[],
            num_dets=0,
            num_ids=0,
            scores=None
        )

        num_dets = 0
        unique_ids = []

        if is_gt:
            headers = ["Frame", "Id", "X", "Y",
                       "Width", "Height", "Score", "Label", "Unused"]
        else:
            headers = ["Frame", "Id", "X", "Y",
                       "Width", "Height", "Score", "Unused_1", "Unused_2", "Unused_3"]

        # data = pd.read_csv(file, names=headers)
        if self._check_header(file):
            data = pd.read_csv(file, skiprows=0, names=headers)
        else:
            data = pd.read_csv(file, names=headers)

        data["Id"] = data["Id"].astype(int)
        data["Score"] = data["Score"].astype(int)
        if is_gt:
            data["Label"] = data["Label"].astype(int)

        data = data.loc[(data["Score"] != 0)]
        if is_gt:
            data = data.loc[(data["Label"].isin(self.__valid_labels))]

        s = data.groupby("Frame")
        ids = s["Id"].apply(lambda x: x)
        bboxs = s[["X", "Y", "Width", "Height"]].apply(lambda x: x)

        for i in s.indices.keys():
            dets_arr = bboxs.iloc[s.indices[i]].values
            ids_arr = ids.iloc[s.indices[i]].values
            num_dets += dets_arr.shape[0]

            unique_ids += list(np.unique(ids_arr))

            results_dict["dets"].append(dets_arr)
            results_dict["ids"].append(ids_arr)

        # Re-label IDs such that there are no empty IDs
        if len(unique_ids) > 0:
            unique_ids = np.unique(unique_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_ids) + 1))
            gt_id_map[unique_ids] = np.arange(len(unique_ids))
            for t in range(len(s.indices)):
                if len(results_dict["ids"][t]) > 0:
                    results_dict["ids"][t] = gt_id_map[results_dict["ids"][t]].astype(np.int)

        results_dict["num_dets"] = num_dets
        results_dict["num_ids"] = len(unique_ids)

        tracking_data = TrackingObject(**results_dict)

        return tracking_data

import json
from operator import itemgetter

import numpy as np
from cv_eval_metrics.base import BaseDataset
from cv_eval_metrics.objects.detection_info import DetectionObject


class COCO(BaseDataset):
    def __init__(self, gt_path: str, pred_path: str, file_format: str = None) -> None:
        super().__init__(gt_path, pred_path, file_format)
        self._gt_path = gt_path
        self._pred_path = pred_path
        self._file_format = file_format
        self.lbl = []

        self._gt_dataset = self._load_json(self._gt_path)
        self._pred_dataset = self._load_json(self._pred_path)

        # self._gt_dataset = self._preprocessed_coco_gt(self._gt_dataset)

    def _load_json(self, file: str) -> dict:
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    def _preprocessed_coco_gt(gt_dataset: dict, image_id_limit: int = 1292):
        gt_dataset.pop("info", None)
        gt_dataset.pop("licenses", None)
        for img in gt_dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)

        gt = []
        if "annotations" in gt_dataset:
            for ann in gt_dataset["annotations"]:
                if ann["image_id"] > image_id_limit:
                    continue
                ann_dict = dict(
                    image_id=ann["image_id"],
                    category_id=ann["category_id"],
                    bbox=ann["bbox"]
                )
                gt.append(ann_dict)
        gt = sorted(gt, key=itemgetter('image_id'))
        return gt

    def process(self, gt_file: str = None, pred_file: str = None):
        gt_data = self._convert_data(self._gt_dataset, is_gt=True)
        pred_data = self._convert_data(self._pred_dataset, is_gt=False)

        return gt_data, pred_data

    def _convert_data(self, json_data: dict, is_gt: bool):
        data = {}
        scores = None
        for ann in json_data:
            image_id = ann["image_id"]
            if image_id in data:
                data[image_id]["bboxes"].append(ann['bbox'])
                data[image_id]["labels"].append(ann['category_id'])
                if not is_gt:
                    data[image_id]["scores"].append(ann['score'])
            else:
                data[image_id] = {
                    'bboxes': [ann['bbox']],
                    'labels': [ann['category_id']],
                    'scores': [ann['score']] if not is_gt else None
                }

        bboxes = [np.array(data[i]['bboxes']) for i in data]
        labels = [np.array(data[i]['labels']) for i in data]
        if not is_gt:
            scores = [np.array(data[i]['scores']) for i in data]

        detection_data = DetectionObject(bboxes=bboxes, labels=labels, scores=scores)
        return detection_data

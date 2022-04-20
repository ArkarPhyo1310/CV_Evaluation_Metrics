from typing import Any, List, Tuple

import numpy as np
from cv_eval_metrics.utils.check import is_float_list, is_int_list, is_str_list


def calculate_base(
    pred: np.ndarray,
    gt: np.ndarray,
    multi_label: bool = False,
    mdmc: str = "global",
    average: str = "micro"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if multi_label and mdmc == "global":
        pred = np.transpose(pred, (0, 2, 1)).reshape(-1, pred.shape[1])
        gt = np.transpose(gt, (0, 2, 1)).reshape(-1, gt.shape[1])

    true_pred, false_pred = gt == pred, gt != pred
    pos_pred, neg_pred = pred == 1, pred == 0

    if average == "micro":
        dim = (0, 1) if pred.ndim == 2 else (1, 2)
    elif average == "macro":
        dim = 0 if pred.ndim == 2 else 2

    tp = np.sum(true_pred * pos_pred, axis=dim)
    fp = np.sum(false_pred * pos_pred, axis=dim)

    tn = np.sum(true_pred * neg_pred, axis=dim)
    fn = np.sum(false_pred * neg_pred, axis=dim)

    return tp, fp, tn, fn


def to_categorical(cls_list: List[str], cls_names: List[str]) -> np.ndarray:
    categorical_cls = []
    for name in cls_names:
        idx = cls_list.index(name)
        categorical_cls.append(idx)

    return np.array(categorical_cls, dtype=int)


def to_onehot(arr: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[arr]


def choose_topk(pred: np.ndarray, top_k: int, multi_label: bool = False):
    if pred.dtype != np.float32:
        raise TypeError("Predictions must have floating point scores.")
    if pred.ndim == 2:
        sorted_pred = np.argsort(pred, kind='mergesort')[:, ::-1]
        s_pred = sorted_pred[:, :top_k]
    elif pred.ndim == 3:
        sorted_pred = np.argsort(pred, kind='mergesort')[:, :, ::-1]
        s_pred = sorted_pred[:, :, :top_k]

    pred_one_hot = np.zeros_like(pred)

    axis = 1
    if multi_label:
        axis = 2

    np.put_along_axis(pred_one_hot, indices=s_pred, values=1, axis=axis)

    return pred_one_hot


def input_format_classification(classes: list, l: Any) -> np.ndarray:
    if is_int_list(l):
        data = np.array(l, dtype=int)
    elif is_str_list(l):
        data = to_categorical(classes, l)
    elif is_float_list(l):
        data = np.array(l, dtype=np.float32)
    elif isinstance(l, np.ndarray):
        data = l.astype(np.float32)
    else:
        raise ValueError("Invalid input value!")

    return data


def prob_to_cls(prob: np.ndarray) -> np.ndarray:
    return np.argmax(prob, axis=-1).astype(int)

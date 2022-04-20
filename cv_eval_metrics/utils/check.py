from typing import Any, Dict, List, Tuple


def is_int_list(l: List[Any]) -> bool:
    return any(isinstance(v, int) for v in l)


def is_str_list(l: List[Any]) -> bool:
    return any(isinstance(v, str) for v in l)


def is_float_list(l: List[Any]) -> bool:
    depth = check_list_depth(l)
    if depth == 2:
        return any(isinstance(v, float) for second in l for v in second)
    elif depth == 3:
        return any(isinstance(v, float) for second in l for third in second for v in third)


def check_list_depth(lst):
    d = 0
    for item in lst:
        if isinstance(item, list):
            d = max(check_list_depth(item), d)
    return d + 1

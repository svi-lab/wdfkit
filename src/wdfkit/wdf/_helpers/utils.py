# -*- coding: utf-8 -*-
"""Small helpers shared by the WDF reader (time, formatting, arrays)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np


def convert_time(t):
    """Convert the Windows 64bit timestamp to the human-readable format.

    Input:
    -------
        t: int or array of ints
        it is the number of microseconds from _EPOCH
    Output:
    -------
        datetime in unix (python) mode
    """
    _epoch = datetime(year=1601, month=1, day=1, tzinfo=timezone.utc)
    return _epoch + t * timedelta(microseconds=1)


def hr_filesize(filesize, suffix="B"):
    """Transform the filesize into Human-readable format."""

    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(filesize) < 1024.0:
            return f"{filesize:3.1f}{unit}{suffix}"
        filesize /= 1024.0
    return f"{filesize:.1f}Yi{suffix}"


def pad_if_unfinished(arr, count, capacity, replace_value=None, extend=False):
    """If the measurement was unfinished, fills the missing values.

    Parameters:
    -----------
    arr: numpy array
    count: int
        the number of recorded values
    capacity: int
        The total number of values expected
    replace value: float or array
        How to fill the missing values. If array is provided, it must
        be of length `capacity - count`.
        The default is to continue filling with the same increment as in
        ``arr[:count]``.
    extend: bool
        Default is False
        If True, `replace_value` is ignored, and the array is extended using
        its mean increment value as a step to produce subsequent missing
        values.

    Returns:
    --------
    The updated array of length = `capacity`,
    with the values arr[count:] equal to replace_value

    Note:
    -----
    It would be useful to have the option to extend the initial array
    so that it continues its evolution.
    In case of time, for example, it may be something like:
    ```python
    interval = np.mean(np.diff(input_array)) * np.ones(capacity - count + 1)
    missing_values_arr = input_array[-1] + np.cumsum(interval[1:])
    output_arr = np.concatenate([input_array, missing_values_arr]`
    ```
    """

    if count < capacity:
        if extend:
            meandiff = np.mean(np.diff(arr[:count]))
            increment_arr = meandiff * np.ones(capacity - count)
            replace_value = arr[count - 1] + np.cumsum(increment_arr)
        if len(arr) == capacity:
            arr[count:] = replace_value
        elif len(arr) == count:
            arr = np.concatenate([arr, replace_value])

    return arr

#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
from einops import rearrange
from gluonts.time_feature import norm_freq_str
from jaxtyping import Num

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


class PatchSizeConstraints(abc.ABC):
    @abc.abstractmethod
    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]: ...

    def __call__(self, freq: str) -> range:
        # to_offset函数从字符串或datetime.timedelta对象返回DateOffset对象。
        # 如：to_offset("5min") = <5 * Minutes>；to_offset("1D1h") = <25 * Hours>
        # to_offset("2W") = <2 * Weeks: weekday=6>；to_offset("2B") = <2 * BusinessDays>
        offset = pd.tseries.frequencies.to_offset(freq)
        # norm_freq_str函数从输入中获得base_freq
        # _get_boundaries获得patch大小的上下界？？
        start, stop = self._get_boundaries(offset.n, norm_freq_str(offset.name))
        # 返回start到stop的范围
        return range(start, stop + 1)


@dataclass
class FixedPatchSizeConstraints(PatchSizeConstraints):
    start: int
    stop: Optional[int] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = self.start
        assert self.start <= self.stop

    # 继承后完成_get_boundaries函数
    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        return self.start, self.stop


# 默认的patch大小？
class DefaultPatchSizeConstraints(PatchSizeConstraints):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    DEFAULT_RANGES = {
        "S": (64, 128),  # 512s = 8.53min, 4096s = 68.26min
        "T": (32, 128),  # 64min = 1.07h, 512min = 8.53h
        "H": (32, 64),  # 128h = 5.33days
        "D": (16, 32),
        "B": (16, 32),
        "W": (16, 32),
        "M": (8, 32),
        "Q": (1, 8),
        "Y": (1, 8),
        "A": (1, 8),
    }

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        start, stop = self.DEFAULT_RANGES[offset_name]
        return start, stop


# 这里才是获取patch_size大小的地方。
# 一共有8-128共5种规格
@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    target_field: str = "target"
    patch_sizes: tuple[int, ...] | list[int] | range = (8, 16, 32, 64, 128)
    patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()
    offset: bool = True

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # 获得当前数据的freq，并根据该freq得到默认的patch_size的上下界
        freq = data_entry["freq"]
        constraints = self.patch_size_constraints(freq)
        # largest patch size based on min_time_patches
        target: list[UnivarTimeSeries] = data_entry[self.target_field]
        length = target[0].shape[0]
        
        # 数据长度除以最少需要的patch个数，得到patch_size的上界
        patch_size_ceil = length // self.min_time_patches

        # patch大小必须同时在patch_sizes范围中做挑选，又不能超出constraints的上下界
        if isinstance(self.patch_sizes, (tuple, list)):
            patch_size_candidates = [
                patch_size
                for patch_size in self.patch_sizes
                if (patch_size in constraints) and (patch_size <= patch_size_ceil)
            ]
        elif isinstance(self.patch_sizes, range):
            patch_size_candidates = range(
                max(self.patch_sizes.start, constraints.start),
                min(self.patch_sizes.stop, constraints.stop, patch_size_ceil),
            )
        else:
            raise NotImplementedError

        if len(patch_size_candidates) <= 0:
            ts_shape = (len(target),) + target[0].shape
            raise AssertionError(
                "no valid patch size candidates for "
                f"time series shape: {ts_shape}, "
                f"freq: {freq}, "
                f"patch_sizes: {self.patch_sizes}, "
                f"constraints: {constraints}, "
                f"min_time_patches: {self.min_time_patches}, "
                f"patch_size_ceil: {patch_size_ceil}"
            )

        # 最后从候选中随机抽取一个作为patch_size？？？
        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry


# 做patch的切割！！
@dataclass
class Patchify(MapFuncMixin, Transformation):
    max_patch_size: int
    # 哪个列需要做patching
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)
    pad_value: int | float = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        patch_size = data_entry["patch_size"]
        # 对data_entry中的各个列施加_patchify函数！
        self.map_func(
            partial(self._patchify, patch_size=patch_size),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _patchify(self, data_entry: dict[str, Any], field: str, patch_size: int):
        arr = data_entry[field]
        # 调用_patchify_arr函数，对arr做patchify！
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)

    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        # 需要保证长度能正好被patch_size整除
        assert arr.shape[-1] % patch_size == 0
        # 按照patch大小做rearrange，其实就是做切割！！！
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        # 再做一个padding？
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        # 表示在最开头做0个padding，在末尾做(self.max_patch_size - patch_size)长度的padding
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        # constant表示用常数做padding，常数为self.pad_value，默认为0
        # numpy.pad：https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr

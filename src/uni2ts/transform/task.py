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

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from jaxtyping import Bool, Float

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin


# 预测任务
# 这里有三个点需要采样：
# 1、均匀采样一个总窗口长度，长度范围为[2, 512]，同时划分出预测窗口的比例为[0.15, 0.5]。
# 2、在变量维度上使用二项分布采样（n=128，a=2，b=5），最多支持128个变量，平均约为37个变量。
# 3、将多个单变量时间序列随机组合成多变量。
# 其中1在这里实现，2和3参考resample.py！
@dataclass
class MaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    min_mask_ratio: float
    max_mask_ratio: float
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __post_init__(self):
        assert (
            self.min_mask_ratio <= self.max_mask_ratio
        ), "min_mask_ratio must be <= max_mask_ratio"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        # 先获得对预测窗口的mask
        prediction_mask = self._generate_prediction_mask(target)
        # 然后输入数据删除掉被mask的预测窗口部分？
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        # 同时保留预测窗口的mask标记
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    # 生成预测的mask
    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        # 初始化
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        
        # mask_ratio在上下限之间（如0.15~0.5）随机得到，表示将一部分设置为预测窗口。
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        # mask的长度为time * mask_ratio，但要保证最小长度是1
        mask_length = max(1, round(time * mask_ratio))
        
        # 将最后的mask_length的部分当作预测窗口，其mask被置为True
        prediction_mask[:, -mask_length:] = True
        
        return prediction_mask

    # 对数据(data_entry[field])做窗口上的裁剪
    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        # 获取没有被mask的那一部分数据
        return self._truncate_arr(arr, mask)

    # 只取出没有被mask的那一部分数据？
    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]


# 延长mask？
# 在mask_field基础上再加一个aux的mask；
# 但这里的mask全为0，所以相当于没有加？
@dataclass
class ExtendMask(CheckArrNDimMixin, CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    mask_field: str
    optional_fields: tuple[str, ...] = tuple()
    expected_ndim: int = 2
    
    # PS：mask_field默认为"prediction_mask"
    # fields默认为tuple()
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target_mask: np.ndarray = data_entry[self.mask_field]
        aux_target_mask: list[np.ndarray] = self.collect_func_list(
            self._generate_target_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        # 在mask_field基础上再加一个aux的mask；
        # 但这里的mask全为0，所以相当于没有加？
        data_entry[self.mask_field] = [target_mask] + aux_target_mask
        return data_entry

    def _generate_target_mask(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr: np.ndarray = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        # 先是一个var*time的全为0的mask？
        field_target_mask = np.zeros((var, time), dtype=bool)
        return field_target_mask


@dataclass
class EvalMaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    mask_length: int
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        prediction_mask[:, -self.mask_length :] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]

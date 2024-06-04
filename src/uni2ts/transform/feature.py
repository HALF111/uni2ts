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
from typing import Any

import numpy as np
from einops import repeat

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin


# 生成varitae的id！！
@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = False
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # 计数君起始位0？
        self.counter = 0
        # 如果随机选取，则从[0,1,...,max_dim-1]中采样max_dim个点，组成variate-id_list。并且replace=False表示不可以取相同数字
        # 否则的话就直接用[0,1,...,max_dim-1]
        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )
        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)  # 检查arr的维度应当和expected_ndim相等？
        # 当前数据的维度就是dim*time，所以生成的ids也应当是这个大小？
        dim, time = arr.shape[:2]
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        
        # 生成dim个variate，每个的长度为time
        # varite从上面self.dimensions中获得连续的dim个来得到
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter : self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        # time_id就是用numpy的arange函数生成的[0,1,...,time-1]的序列！！
        field_seq_id = np.arange(time)
        # 然后重复var遍即可
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        # 返回observed_mask
        data_entry[self.observed_mask_field] = observed_mask
        return data_entry

    # NaN的值会变成False，正常值的地方为True
    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)
